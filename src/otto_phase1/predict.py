from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .candidates import build_candidate_pool
from .config import BaselineConfig
from .scoring import score_candidates_for_target
from .types import CoVisitationModel, Event, PopularityModel

TARGETS = ("clicks", "carts", "orders")


def _session_events_from_df(session_df: pd.DataFrame) -> List[Event]:
    return list(
        zip(
            session_df["aid"].astype("int64").tolist(),
            session_df["ts"].astype("int64").tolist(),
            session_df["type"].astype("string").tolist(),
        )
    )


def predict_for_test_shards(
    test_shards,
    popularity: PopularityModel,
    covisitation: CoVisitationModel,
    config: BaselineConfig,
) -> Dict[Tuple[int, str], List[int]]:
    """
    Build predictions for each (session, target) key.
    This stage intentionally does not touch test_labels.
    """
    predictions: Dict[Tuple[int, str], List[int]] = {}

    for shard_df in test_shards:
        for session, session_df in shard_df.groupby("session", sort=False):
            session_events = _session_events_from_df(session_df)
            session = int(session)

            # Shared retrieval, target-specific scoring.
            shared_pool = build_candidate_pool(
                session_events=session_events,
                popularity=popularity,
                covisitation=covisitation,
                config=config,
            )

            for target in TARGETS:
                topk = score_candidates_for_target(
                    session_events=session_events,
                    candidate_pool=shared_pool,
                    popularity=popularity,
                    config=config,
                    target=target,
                )
                predictions[(session, target)] = topk

    return predictions


def predictions_to_submission_df(
    predictions: Dict[Tuple[int, str], List[int]]
) -> pd.DataFrame:
    rows = []
    for (session, target), aids in predictions.items():
        rows.append(
            {
                "session_type": f"{session}_{target}",
                "labels": " ".join(map(str, aids)),
            }
        )
    return pd.DataFrame(rows)
