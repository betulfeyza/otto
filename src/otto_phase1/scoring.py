from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

from .config import BaselineConfig
from .types import CandidatePool, Event, PopularityModel


def _inv_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (1.0 + float(rank))


def compute_candidate_feature_and_score_maps(
    session_events: Sequence[Event],
    candidate_pool: CandidatePool,
    popularity: PopularityModel,
    config: BaselineConfig,
    target: str,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    """
    Build candidate-level features and final heuristic score map for one target.
    The output is shared between inference ranking and training-table generation.
    """
    weights = config.target_weights[target]
    truncated = list(session_events)[-config.session_history_max_events :]

    hist_presence: Set[int] = set()
    hist_click = defaultdict(float)
    hist_cart = defaultdict(float)
    hist_order = defaultdict(float)

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

    feature_map: Dict[int, Dict[str, float]] = {}
    score_map: Dict[int, float] = {}

    for aid in candidate_pool.aids:
        aid = int(aid)

        hist_click_v = hist_click.get(aid, 0.0)
        hist_cart_v = hist_cart.get(aid, 0.0)
        hist_order_v = hist_order.get(aid, 0.0)
        hist_presence_v = 1.0 if aid in hist_presence else 0.0
        covis_v = float(candidate_pool.covis_signal.get(aid, 0.0))
        is_last_item_v = 1.0 if (last_item is not None and aid == last_item) else 0.0

        pop_target_rank = popularity.per_type_rank.get(target, {}).get(aid)
        pop_weighted_rank = popularity.weighted_rank.get(aid)
        pop_global_rank = popularity.global_rank.get(aid)

        pop_target_inv = _inv_rank(pop_target_rank)
        pop_weighted_inv = _inv_rank(pop_weighted_rank)
        pop_global_inv = _inv_rank(pop_global_rank)

        pop_used_inv = pop_target_inv
        if pop_used_inv == 0.0:
            pop_used_inv = pop_weighted_inv
        if pop_used_inv == 0.0:
            pop_used_inv = pop_global_inv

        score = 0.0
        score += weights["hist_click"] * hist_click_v
        score += weights["hist_cart"] * hist_cart_v
        score += weights["hist_order"] * hist_order_v
        score += weights["hist_presence"] * hist_presence_v
        score += weights["covis"] * covis_v
        score += weights["popular"] * pop_used_inv
        if is_last_item_v > 0.0:
            score += weights["last_item_boost"]

        feature_map[aid] = {
            "hist_click_score": hist_click_v,
            "hist_cart_score": hist_cart_v,
            "hist_order_score": hist_order_v,
            "hist_presence": hist_presence_v,
            "covis_signal": covis_v,
            "is_last_item": is_last_item_v,
            "pop_target_inv_rank": pop_target_inv,
            "pop_weighted_inv_rank": pop_weighted_inv,
            "pop_global_inv_rank": pop_global_inv,
            "target_weight_hist_click": float(weights["hist_click"]),
            "target_weight_hist_cart": float(weights["hist_cart"]),
            "target_weight_hist_order": float(weights["hist_order"]),
            "target_weight_hist_presence": float(weights["hist_presence"]),
            "target_weight_covis": float(weights["covis"]),
            "target_weight_popular": float(weights["popular"]),
            "target_weight_last_item": float(weights["last_item_boost"]),
        }
        score_map[aid] = score

    return feature_map, score_map


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
    _, score_map = compute_candidate_feature_and_score_maps(
        session_events=session_events,
        candidate_pool=candidate_pool,
        popularity=popularity,
        config=config,
        target=target,
    )

    scored: List[Tuple[int, float]] = [
        (int(aid), float(score_map[int(aid)])) for aid in candidate_pool.aids
    ]

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
