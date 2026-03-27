from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Set, Tuple

from .config import BaselineConfig
from .types import CandidatePool, Event, PopularityModel


def _inv_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (1.0 + float(rank))


def score_candidates_for_target(
    session_events: Sequence[Event],
    candidate_pool: CandidatePool,
    popularity: PopularityModel,
    config: BaselineConfig,
    target: str,
) -> List[int]:
    """
    Target-specific heuristic scoring on a shared candidate pool.
    """
    weights = config.target_weights[target]
    truncated = list(session_events)[-config.session_history_max_events :]

    hist_presence: Set[int] = set()
    hist_click = defaultdict(float)
    hist_cart = defaultdict(float)
    hist_order = defaultdict(float)

    # Reverse traversal: recent events get stronger decay weight.
    for idx, (aid, _, ev_type) in enumerate(reversed(truncated)):
        aid = int(aid)
        hist_presence.add(aid)
        recency_weight = config.recency_decay**idx

        if ev_type == "clicks":
            hist_click[aid] += recency_weight
        elif ev_type == "carts":
            hist_cart[aid] += recency_weight
        elif ev_type == "orders":
            hist_order[aid] += recency_weight

    last_item = int(truncated[-1][0]) if truncated else None

    scored: List[Tuple[int, float]] = []
    for aid in candidate_pool.aids:
        aid = int(aid)

        score = 0.0
        score += weights["hist_click"] * hist_click.get(aid, 0.0)
        score += weights["hist_cart"] * hist_cart.get(aid, 0.0)
        score += weights["hist_order"] * hist_order.get(aid, 0.0)
        score += weights["hist_presence"] * (1.0 if aid in hist_presence else 0.0)
        score += weights["covis"] * candidate_pool.covis_signal.get(aid, 0.0)

        pop_rank = popularity.per_type_rank.get(target, {}).get(aid)
        if pop_rank is None:
            pop_rank = popularity.weighted_rank.get(aid)
        if pop_rank is None:
            pop_rank = popularity.global_rank.get(aid)
        score += weights["popular"] * _inv_rank(pop_rank)

        if last_item is not None and aid == last_item:
            score += weights["last_item_boost"]

        scored.append((aid, score))

    scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)

    # Ensure uniqueness and cut top-20 (or configured K).
    output: List[int] = []
    seen: Set[int] = set()
    for aid, _ in scored:
        if aid in seen:
            continue
        seen.add(aid)
        output.append(aid)
        if len(output) >= config.topk_per_target:
            break

    return output
