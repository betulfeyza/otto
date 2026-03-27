from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

from .config import BaselineConfig
from .types import CandidatePool, CoVisitationModel, Event, PopularityModel


def _unique_preserve_order(items: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    output: List[int] = []
    for item in items:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def build_candidate_pool(
    session_events: Sequence[Event],
    popularity: PopularityModel,
    covisitation: CoVisitationModel,
    config: BaselineConfig,
) -> CandidatePool:
    """
    Candidate sources (in this order):
    1) recent session history items
    2) co-visitation neighbors
    3) popularity fallback (target-aware + weighted + global)
    """
    truncated = list(session_events)[-config.session_history_max_events :]
    history_aids = [int(aid) for aid, _, _ in truncated]
    history_unique = _unique_preserve_order(reversed(history_aids))

    # co-visitation expansion with additive neighbor scores
    covis_signal: Dict[int, float] = defaultdict(float)
    covis_candidates: List[int] = []

    for aid in history_unique:
        neighbors = covisitation.neighbors.get(aid, [])
        for nb_aid, score in neighbors[: config.covis_candidates_per_item]:
            nb_aid = int(nb_aid)
            covis_signal[nb_aid] += float(score)
            covis_candidates.append(nb_aid)

    # target-agnostic fallback order for shared candidate pool
    fallback = (
        popularity.weighted_top
        + popularity.global_top
        + popularity.per_type_top.get("clicks", [])
        + popularity.per_type_top.get("carts", [])
        + popularity.per_type_top.get("orders", [])
    )

    merged = history_unique + _unique_preserve_order(covis_candidates)

    if len(merged) < config.min_candidates_before_fallback:
        merged = _unique_preserve_order(merged + fallback)

    return CandidatePool(aids=merged, covis_signal=dict(covis_signal))
