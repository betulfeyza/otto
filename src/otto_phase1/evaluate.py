from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import BaselineConfig


TARGETS = ("clicks", "carts", "orders")


def _recall_for_target(
    labels_df: pd.DataFrame,
    predictions: Dict[Tuple[int, str], List[int]],
    target: str,
    topk: int,
) -> float:
    subset = labels_df[labels_df["type"] == target]

    numer = 0.0
    denom = 0.0

    for row in subset.itertuples(index=False):
        session = int(row.session)
        gt = list(row.ground_truth) if row.ground_truth is not None else []
        gt = [int(a) for a in gt]

        pred = predictions.get((session, target), [])[:topk]
        pred_set = set(pred)
        gt_set = set(gt)

        hits = len(pred_set & gt_set)
        numer += float(hits)
        denom += float(min(topk, len(gt_set)))

    return 0.0 if denom == 0 else numer / denom


def evaluate_predictions(
    labels_path,
    predictions: Dict[Tuple[int, str], List[int]],
    config: BaselineConfig,
) -> Dict[str, float]:
    """
    Offline OTTO-style Recall@20:
    - clicks, carts, orders are reported separately
    - weighted final score is returned using config.metric_weights
    This function is the only place where test_labels is read.
    """
    labels_df = pd.read_parquet(labels_path)

    required = {"session", "type", "ground_truth"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"test_labels is missing columns: {missing}")

    metrics: Dict[str, float] = {}
    for target in TARGETS:
        metrics[f"recall@{config.topk_per_target}_{target}"] = _recall_for_target(
            labels_df=labels_df,
            predictions=predictions,
            target=target,
            topk=config.topk_per_target,
        )

    weighted = 0.0
    for target in TARGETS:
        weighted += (
            config.metric_weights[target]
            * metrics[f"recall@{config.topk_per_target}_{target}"]
        )
    metrics["weighted_score"] = weighted

    return metrics


def evaluate_submission_file(
    labels_path: str | Path,
    submission_csv_path: str | Path,
    topk: int = 20,
) -> Dict[str, float]:
    """
    Evaluate a submission CSV file directly without re-running the pipeline.
    
    Args:
        labels_path: path to test_labels.parquet
        submission_csv_path: path to submission CSV (session_type, labels columns)
        topk: number of top items to consider (default 20)
    
    Returns:
        dict with recall@20_{clicks,carts,orders} and weighted_score
    
    Submission CSV format: session_type,labels
    - session_type: "{session_id}_{target}" e.g. "12906577_clicks"
    - labels: space-delimited aid list e.g. "123 456 789"
    """
    labels_path = Path(labels_path)
    submission_csv_path = Path(submission_csv_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    if not submission_csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {submission_csv_path}")

    # Parse submission CSV into predictions dict
    submission_df = pd.read_csv(submission_csv_path)
    required_cols = {"session_type", "labels"}
    missing_cols = required_cols - set(submission_df.columns)
    if missing_cols:
        raise ValueError(f"Submission CSV missing columns: {missing_cols}")

    predictions: Dict[Tuple[int, str], List[int]] = {}

    for row in submission_df.itertuples(index=False):
        session_type_str = str(row.session_type).strip()
        labels_str = str(row.labels).strip() if row.labels else ""

        # Parse session_type: "{session}_{target}"
        parts = session_type_str.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid session_type format: {session_type_str}")

        try:
            session = int(parts[0])
        except ValueError:
            raise ValueError(f"Could not parse session ID from: {parts[0]}")

        target = parts[1].lower()
        if target not in TARGETS:
            raise ValueError(f"Invalid target type: {target}")

        # Parse labels: space-delimited aid list
        aids: List[int] = []
        seen: set = set()
        if labels_str:
            for aid_str in labels_str.split():
                try:
                    aid = int(aid_str)
                except ValueError:
                    raise ValueError(f"Could not parse aid from: {aid_str}")

                # Remove duplicates while preserving order
                if aid not in seen:
                    aids.append(aid)
                    seen.add(aid)

        # Enforce topk
        aids = aids[:topk]
        predictions[(session, target)] = aids

    # Load ground truth labels
    labels_df = pd.read_parquet(labels_path)
    required_label_cols = {"session", "type", "ground_truth"}
    missing_label_cols = required_label_cols - set(labels_df.columns)
    if missing_label_cols:
        raise ValueError(f"test_labels missing columns: {missing_label_cols}")

    # Compute recall per target
    metric_weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    metrics: Dict[str, float] = {}

    for target in TARGETS:
        numer = 0.0
        denom = 0.0

        subset = labels_df[labels_df["type"] == target]

        for row in subset.itertuples(index=False):
            session = int(row.session)
            gt = row.ground_truth
            
            # Handle various ground_truth formats (numpy array, list, scalar, None)
            if gt is None or (isinstance(gt, float) and pd.isna(gt)):
                gt_list = []
            elif hasattr(gt, "tolist"):  # numpy array
                gt_list = [int(a) for a in gt.tolist()]
            elif isinstance(gt, list):
                gt_list = [int(a) for a in gt]
            elif isinstance(gt, str):
                # Handle string representation of list
                try:
                    gt_list = [int(a) for a in gt.strip("[]").split(",")]
                except (ValueError, AttributeError):
                    gt_list = []
            else:
                # Scalar value
                try:
                    gt_list = [int(gt)]
                except (ValueError, TypeError):
                    gt_list = []

            pred_list = predictions.get((session, target), [])[:topk]
            pred_set = set(pred_list)
            gt_set = set(gt_list)

            hits = len(pred_set & gt_set)
            numer += float(hits)
            denom += float(min(topk, len(gt_set)))

        recall = 0.0 if denom == 0 else numer / denom
        metrics[f"recall@{topk}_{target}"] = recall

    # Compute weighted score
    weighted = sum(
        metric_weights[target] * metrics[f"recall@{topk}_{target}"]
        for target in TARGETS
    )
    metrics["weighted_score"] = weighted

    return metrics
