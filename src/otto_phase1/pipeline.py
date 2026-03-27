from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import BaselineConfig
from .covisitation import build_covisitation_from_train
from .evaluate import evaluate_predictions
from .io_utils import iter_shard_dataframes
from .popularity import build_popularity_from_train
from .predict import predict_for_test_shards, predictions_to_submission_df


def run_phase1_pipeline(
    config: BaselineConfig,
    save_submission_to: Path | None = None,
    run_evaluation: bool = True,
) -> Dict[str, float]:
    """
    End-to-end phase-1 baseline pipeline:
    1) read all train/test shards (sorted by session, ts)
    2) build popularity (train only)
    3) build co-visitation (train only)
    4) candidate generation + heuristic scoring (test sessions)
    5) optional offline evaluation with test_labels (evaluation only)
    """

    # Two-pass train scan to keep memory usage bounded.
    popularity = build_popularity_from_train(
        train_shards=iter_shard_dataframes(config.train_path),
        config=config,
    )
    covis = build_covisitation_from_train(
        train_shards=iter_shard_dataframes(config.train_path),
        config=config,
    )

    test_shards = iter_shard_dataframes(config.test_path)
    predictions = predict_for_test_shards(
        test_shards=test_shards,
        popularity=popularity,
        covisitation=covis,
        config=config,
    )

    if save_submission_to is not None:
        submission_df = predictions_to_submission_df(predictions)
        save_submission_to.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(save_submission_to, index=False)

    if run_evaluation:
        if not config.labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {config.labels_path}")
        return evaluate_predictions(
            labels_path=config.labels_path,
            predictions=predictions,
            config=config,
        )

    return {}
