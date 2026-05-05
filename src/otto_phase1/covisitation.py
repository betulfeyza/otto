from __future__ import annotations

from collections import Counter, defaultdict
from heapq import nlargest
from typing import DefaultDict, Dict, Iterable, List, Tuple

import pandas as pd

from .config import BaselineConfig
from .types import CoVisitationModel, Event


def _prune_counter(counter: Counter[int], keep_top_n: int) -> Counter[int]:
    if len(counter) <= keep_top_n:
        return counter
    return Counter(dict(nlargest(keep_top_n, counter.items(), key=lambda x: x[1])))


def _prune_covis_map(
    covis_map: DefaultDict[int, Counter[int]],
    keep_per_item: int,
) -> None:
    for aid in list(covis_map.keys()):
        covis_map[aid] = _prune_counter(covis_map[aid], keep_per_item)


def _session_pairs(events: List[Event], config: BaselineConfig) -> List[Tuple[int, int]]:
    """
    Generate bounded item-item pairs from one session.
    We limit by:
    - last N events
    - max future neighbors per anchor
    - max time window
    """
    truncated = events[-config.session_max_events_for_covis :]
    n = len(truncated)
    pairs: List[Tuple[int, int]] = []

    for i in range(n):
        aid_i, ts_i, _ = truncated[i]
        max_j = min(n, i + 1 + config.covis_max_pairs_per_anchor)

        for j in range(i + 1, max_j):
            aid_j, ts_j, _ = truncated[j]
            if ts_j - ts_i > config.covis_max_time_window_ms:
                break
            if aid_i == aid_j:
                continue
            pairs.append((aid_i, aid_j))
            pairs.append((aid_j, aid_i))

    return pairs


def build_covisitation_from_train(
    train_shards: Iterable[pd.DataFrame],
    config: BaselineConfig,
    max_ts: int | None = None,
    cutoff_ts: int | None = None,
    train_prefix_ratio: float = 0.8,
) -> CoVisitationModel:
    """
    Build a memory-aware co-visitation graph from train only.
    Pairs are aggregated shard-by-shard with periodic top-K pruning.
    
    If cutoff_ts is provided, only include events from sessions where
    session.max_ts <= cutoff_ts (excludes validation sessions).
    Only the prefix fraction (train_prefix_ratio) of training session events
    are included to prevent leakage from the session-local future portion.
    """
    covis_map: DefaultDict[int, Counter[int]] = defaultdict(Counter)
    keep_per_item = config.covis_topk_neighbors * config.covis_prune_keep_multiplier

    for shard_df in train_shards:
        # Important: decide train sessions from the untruncated shard to avoid
        # accidentally reclassifying valid-crossing sessions as train sessions.
        if cutoff_ts is not None:
            session_max_ts = shard_df.groupby("session")["ts"].max()
            train_sessions = session_max_ts[session_max_ts <= int(cutoff_ts)].index
            shard_df = shard_df[shard_df["session"].isin(train_sessions)]
            if shard_df.empty:
                continue

            # For training sessions, keep only prefix_ratio fraction to avoid
            # leakage from session-local future events to global statistics.
            prefixed_rows = []
            for _, session_df in shard_df.groupby("session", sort=False):
                n = len(session_df)
                split_idx = int(n * train_prefix_ratio)
                split_idx = max(1, min(split_idx, n - 1))
                prefixed_rows.append(session_df.iloc[:split_idx])
            shard_df = (
                pd.concat(prefixed_rows, ignore_index=True) if prefixed_rows else pd.DataFrame()
            )

        if max_ts is not None:
            shard_df = shard_df[shard_df["ts"] <= int(max_ts)]
            if shard_df.empty:
                continue

        for _, session_df in shard_df.groupby("session", sort=False):
            events = list(
                zip(
                    session_df["aid"].tolist(),
                    session_df["ts"].tolist(),
                    session_df["type"].astype("string").tolist(),
                )
            )
            for src_aid, dst_aid in _session_pairs(events, config):
                covis_map[int(src_aid)][int(dst_aid)] += 1

        # Periodic pruning after each shard to keep RAM bounded.
        _prune_covis_map(covis_map, keep_per_item)

    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for src_aid, dst_counter in covis_map.items():
        top_neighbors = nlargest(
            config.covis_topk_neighbors, dst_counter.items(), key=lambda x: x[1]
        )
        neighbors[src_aid] = [(int(aid), float(score)) for aid, score in top_neighbors]

    return CoVisitationModel(neighbors=neighbors)
