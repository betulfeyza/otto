from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


Event = Tuple[int, int, str]  # (aid, ts, type)


@dataclass
class PopularityModel:
    global_top: List[int]
    weighted_top: List[int]
    per_type_top: Dict[str, List[int]]
    global_rank: Dict[int, int]
    weighted_rank: Dict[int, int]
    per_type_rank: Dict[str, Dict[int, int]]


@dataclass
class CoVisitationModel:
    # aid -> list[(neighbor_aid, score)] sorted by score desc
    neighbors: Dict[int, List[Tuple[int, float]]]


@dataclass
class CandidatePool:
    # Unique pool in deterministic order (history -> covis -> fallback)
    aids: List[int]
    # signal from candidate generation, mostly from co-visitation
    covis_signal: Dict[int, float]
