from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .candidates import build_candidate_pool
from .config import BaselineConfig
from .predict import TARGETS
from .scoring import compute_candidate_feature_and_score_maps
from .types import CandidatePool, CoVisitationModel, Event, PopularityModel


_CANDIDATE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "hist_click_score",
    "hist_cart_score",
    "hist_order_score",
    "hist_presence",
    "covis_signal",
    "is_last_item",
    "pop_target_inv_rank",
    "pop_weighted_inv_rank",
    "pop_global_inv_rank",
    "target_weight_hist_click",
    "target_weight_hist_cart",
    "target_weight_hist_order",
    "target_weight_hist_presence",
    "target_weight_covis",
    "target_weight_popular",
    "target_weight_last_item",
)


def _unique_preserve_order(items: Sequence[int]) -> List[int]:
    seen = set()
    output: List[int] = []
    for item in items:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def infer_global_cutoff_ts_from_max_ts(
    train_shards: Iterable[pd.DataFrame],
    holdout_days: int,
) -> int:
    """
    Infer a global time cutoff for validation from max train timestamp.

    validation cutoff = max_ts - holdout_days
    """
    max_ts = None
    holdout_ms = int(holdout_days) * 24 * 60 * 60 * 1000

    for shard_df in train_shards:
        shard_max = int(shard_df["ts"].max())
        max_ts = shard_max if max_ts is None else max(max_ts, shard_max)

    if max_ts is None:
        raise ValueError("Train shards are empty, cannot infer cutoff")

    return int(max_ts - holdout_ms)


def _session_events_from_df(session_df: pd.DataFrame) -> List[Event]:
    return list(
        zip(
            session_df["aid"].astype("int64").tolist(),
            session_df["ts"].astype("int64").tolist(),
            session_df["type"].astype("string").tolist(),
        )
    )


def _split_events_for_valid(events: Sequence[Event], cutoff_ts: int) -> Tuple[List[Event], List[Event]]:
    prefix = [e for e in events if int(e[1]) <= cutoff_ts]
    future = [e for e in events if int(e[1]) > cutoff_ts]
    return prefix, future


def _split_events_for_train(events: Sequence[Event], prefix_ratio: float) -> Tuple[List[Event], List[Event]]:
    if len(events) < 2:
        return [], []

    ratio = float(prefix_ratio)
    ratio = 0.5 if ratio <= 0.0 or ratio >= 1.0 else ratio

    split_idx = int(len(events) * ratio)
    split_idx = max(1, min(split_idx, len(events) - 1))

    return list(events[:split_idx]), list(events[split_idx:])


def _future_ground_truth_sets(future_events: Sequence[Event]) -> Dict[str, set[int]]:
    gt: Dict[str, set[int]] = {"clicks": set(), "carts": set(), "orders": set()}
    for aid, _, ev_type in future_events:
        ev_type = str(ev_type)
        if ev_type in gt:
            gt[ev_type].add(int(aid))
    return gt


def _write_part(rows: List[dict], output_dir: Path, part_idx: int) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    part_path = output_dir / f"part-{part_idx:05d}.parquet"
    pd.DataFrame(rows).to_parquet(part_path, index=False)


def build_candidate_level_training_table(
    train_shards: Iterable[pd.DataFrame],
    popularity: PopularityModel,
    covisitation: CoVisitationModel,
    config: BaselineConfig,
    output_dir: Path,
    cutoff_ts: int,
    train_prefix_ratio: float = 0.8,
    max_candidates_per_target: int = 200,
    max_negatives_per_target: int = 100,
    rows_per_part: int = 600_000,
    gt_injection_mode: str = "train",
) -> Dict[str, int]:
    """
    Build candidate-level feature table with labels for model training.

    Strategy:
    - Validation split (time-based): for sessions crossing cutoff_ts,
      prefix <= cutoff and future > cutoff.
    - Train split (session-local): for sessions ending before cutoff,
      split each session with train_prefix_ratio.
    - Output is written as partitioned parquet parts to stay memory-safe.
    """
    if rows_per_part <= 0:
        raise ValueError("rows_per_part must be > 0")

    mode = str(gt_injection_mode).strip().lower()
    if mode not in {"none", "train", "both"}:
        raise ValueError("gt_injection_mode must be one of: none, train, both")

    part_rows: List[dict] = []
    part_idx = 0

    stats = defaultdict(int)

    for shard_df in train_shards:
        for session, session_df in shard_df.groupby("session", sort=False):
            events = _session_events_from_df(session_df)
            if len(events) < 2:
                continue

            min_ts = int(events[0][1])
            max_ts = int(events[-1][1])

            if min_ts <= cutoff_ts < max_ts:
                split_name = "valid"
                prefix_events, future_events = _split_events_for_valid(events, cutoff_ts)
            elif max_ts <= cutoff_ts:
                split_name = "train"
                prefix_events, future_events = _split_events_for_train(
                    events=events,
                    prefix_ratio=train_prefix_ratio,
                )
            else:
                # Session is fully after cutoff; skip for strict temporal setup.
                continue

            if not prefix_events or not future_events:
                continue

            gt_sets = _future_ground_truth_sets(future_events)
            if all(len(gt_sets[t]) == 0 for t in TARGETS):
                continue

            session = int(session)
            shared_pool = build_candidate_pool(
                session_events=prefix_events,
                popularity=popularity,
                covisitation=covisitation,
                config=config,
            )

            for target in TARGETS:
                gt_target = gt_sets[target]
                if not gt_target:
                    continue

                inject_gt = mode == "both" or (mode == "train" and split_name == "train")

                # Optional GT injection (typically train-only).
                if inject_gt:
                    augmented_aids = _unique_preserve_order(
                        list(shared_pool.aids) + [int(aid) for aid in gt_target]
                    )
                else:
                    augmented_aids = _unique_preserve_order(list(shared_pool.aids))

                augmented_covis_signal = dict(shared_pool.covis_signal)
                if inject_gt:
                    for aid in gt_target:
                        augmented_covis_signal.setdefault(int(aid), 0.0)
                augmented_pool = CandidatePool(
                    aids=augmented_aids,
                    covis_signal=augmented_covis_signal,
                )

                feature_map, score_map = compute_candidate_feature_and_score_maps(
                    session_events=prefix_events,
                    candidate_pool=augmented_pool,
                    popularity=popularity,
                    config=config,
                    target=target,
                )

                ordered_by_score = sorted(
                    score_map.keys(), key=lambda aid: score_map[aid], reverse=True
                )
                ordered_by_score = ordered_by_score[:max_candidates_per_target]

                # If GT injection is disabled for this split, keep evaluation realistic.
                if inject_gt:
                    with_gt = _unique_preserve_order(ordered_by_score + list(gt_target))
                else:
                    with_gt = _unique_preserve_order(ordered_by_score)

                positives = [aid for aid in with_gt if aid in gt_target]
                if not positives:
                    stats[f"{split_name}_session_target_without_positive_rows"] += 1
                    continue

                negatives = [aid for aid in with_gt if aid not in gt_target]
                negatives = negatives[:max_negatives_per_target]

                selected = positives + negatives
                rank_map = {aid: rank for rank, aid in enumerate(with_gt, start=1)}

                stats[f"{split_name}_session_target_rows"] += 1
                stats[f"{split_name}_positives"] += len(positives)
                stats[f"{split_name}_negatives"] += len(negatives)

                for aid in selected:
                    raw_feats = feature_map.get(int(aid), {})
                    # Keep feature matrix dense to avoid NaN-pattern leakage.
                    feats = {
                        name: float(raw_feats.get(name, 0.0))
                        for name in _CANDIDATE_FEATURE_COLUMNS
                    }
                    part_rows.append(
                        {
                            "split": split_name,
                            "session": session,
                            "target": target,
                            "aid": int(aid),
                            "label": int(int(aid) in gt_target),
                            "candidate_rank": int(rank_map.get(int(aid), 999999)),
                            "heuristic_score": float(score_map.get(int(aid), 0.0)),
                            "prefix_len": int(len(prefix_events)),
                            "future_len": int(len(future_events)),
                            "cutoff_ts": int(cutoff_ts),
                            **feats,
                        }
                    )

            if len(part_rows) >= rows_per_part:
                _write_part(part_rows, output_dir=output_dir, part_idx=part_idx)
                stats["parts_written"] += 1
                stats["rows_written"] += len(part_rows)
                part_rows = []
                part_idx += 1

    if part_rows:
        _write_part(part_rows, output_dir=output_dir, part_idx=part_idx)
        stats["parts_written"] += 1
        stats["rows_written"] += len(part_rows)

    return {k: int(v) for k, v in stats.items()}
