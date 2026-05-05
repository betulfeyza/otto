from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import pandas as pd

from .config import BaselineConfig
from .types import PopularityModel


def _ranks_from_list(items: List[int]) -> Dict[int, int]:
    return {aid: rank for rank, aid in enumerate(items, start=1)}


def build_popularity_from_train(
    train_shards: Iterable[pd.DataFrame],
    config: BaselineConfig,
    max_ts: int | None = None,
    cutoff_ts: int | None = None,
    train_prefix_ratio: float = 0.8,
) -> PopularityModel:
    """
    Build global and type-aware popularity lists from train only.
    No test labels are touched here.
    
    If cutoff_ts is provided, only include events from sessions where
    session.max_ts <= cutoff_ts (excludes validation sessions).
    Only the prefix fraction (train_prefix_ratio) of training session events
    are included to prevent leakage from the session-local future portion.
    """
    global_counts: Counter[int] = Counter()
    weighted_counts: Counter[int] = Counter()
    per_type_counts: Dict[str, Counter[int]] = defaultdict(Counter)

    weight_map = config.popularity_event_weights

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

        aid_counts = shard_df["aid"].value_counts()
        global_counts.update(aid_counts.to_dict())

        for event_type, group in shard_df.groupby("type", observed=True):
            event_type = str(event_type)
            group_counts = group["aid"].value_counts().to_dict()
            per_type_counts[event_type].update(group_counts)

            event_weight = float(weight_map.get(event_type, 1.0))
            for aid, cnt in group_counts.items():
                weighted_counts[int(aid)] += float(cnt) * event_weight

    global_top = [aid for aid, _ in global_counts.most_common(config.popularity_topk)]
    weighted_top = [aid for aid, _ in weighted_counts.most_common(config.popularity_topk)]

    per_type_top: Dict[str, List[int]] = {}
    for event_type in ("clicks", "carts", "orders"):
        per_type_top[event_type] = [
            aid for aid, _ in per_type_counts[event_type].most_common(config.popularity_topk)
        ]

    return PopularityModel(
        global_top=global_top,
        weighted_top=weighted_top,
        per_type_top=per_type_top,
        global_rank=_ranks_from_list(global_top),
        weighted_rank=_ranks_from_list(weighted_top),
        per_type_rank={k: _ranks_from_list(v) for k, v in per_type_top.items()},
    )
