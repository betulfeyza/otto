from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import pandas as pd

from .config import BaselineConfig
from .types import PopularityModel


def _ranks_from_list(items: List[int]) -> Dict[int, int]:
    return {aid: rank for rank, aid in enumerate(items, start=1)}


def build_popularity_from_train(
    train_shards: Iterable[pd.DataFrame], config: BaselineConfig
) -> PopularityModel:
    """
    Build global and type-aware popularity lists from train only.
    No test labels are touched here.
    """
    global_counts: Counter[int] = Counter()
    weighted_counts: Counter[int] = Counter()
    per_type_counts: Dict[str, Counter[int]] = defaultdict(Counter)

    weight_map = config.popularity_event_weights

    for shard_df in train_shards:
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
