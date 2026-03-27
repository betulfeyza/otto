from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class BaselineConfig:
    data_root: Path = Path("archive")
    train_dir: str = "train_parquet"
    test_dir: str = "test_parquet"
    labels_file: str = "test_labels.parquet"
    topk_per_target: int = 20

    # Popularity
    popularity_topk: int = 300
    popularity_event_weights: Dict[str, float] = field(
        default_factory=lambda: {"clicks": 1.0, "carts": 6.0, "orders": 10.0}
    )

    # Co-visitation (memory-aware)
    session_max_events_for_covis: int = 30
    covis_max_time_window_ms: int = 24 * 60 * 60 * 1000
    covis_max_pairs_per_anchor: int = 20
    covis_topk_neighbors: int = 40
    covis_prune_keep_multiplier: int = 4

    # Candidate generation
    session_history_max_events: int = 30
    covis_candidates_per_item: int = 25
    min_candidates_before_fallback: int = 120

    # Heuristic scoring weights
    recency_decay: float = 0.92
    target_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "clicks": {
                "hist_click": 2.5,
                "hist_cart": 1.0,
                "hist_order": 0.8,
                "hist_presence": 0.8,
                "last_item_boost": 2.0,
                "covis": 1.8,
                "popular": 0.7,
            },
            "carts": {
                "hist_click": 0.8,
                "hist_cart": 2.4,
                "hist_order": 1.8,
                "hist_presence": 1.0,
                "last_item_boost": 1.2,
                "covis": 1.6,
                "popular": 0.9,
            },
            "orders": {
                "hist_click": 0.6,
                "hist_cart": 1.8,
                "hist_order": 2.8,
                "hist_presence": 1.2,
                "last_item_boost": 1.0,
                "covis": 1.4,
                "popular": 1.1,
            },
        }
    )

    # Offline evaluation (OTTO metric)
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    )

    @property
    def train_path(self) -> Path:
        return self.data_root / self.train_dir

    @property
    def test_path(self) -> Path:
        return self.data_root / self.test_dir

    @property
    def labels_path(self) -> Path:
        return self.data_root / self.labels_file
