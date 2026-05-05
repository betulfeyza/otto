#!/usr/bin/env python3
"""
Train simple Neural Network (MLP) models on candidate-level training data.

This script trains one binary MLP classifier per target:
- clicks
- carts
- orders

Inputs are the same 16 engineered features used in tree-based baselines.
Outputs: pickled sklearn pipelines and validation metrics with Recall@20.
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "hist_click_score",
    "hist_cart_score",
    "hist_order_score",
    "hist_presence",
    "covis_signal",
    "is_last_item",
    "pop_target_inv_rank",
    "pop_weighted_inv_rank",
    "pop_global_inv_rank",
    "target_weight_hist_click",
    "target_weight_hist_cart",
    "target_weight_hist_order",
    "target_weight_hist_presence",
    "target_weight_covis",
    "target_weight_popular",
    "target_weight_last_item",
]


def load_training_table(
    table_dir: Path,
    max_parts: int | None = None,
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Load all parquet parts from a training table directory."""
    print(f"Loading training table from {table_dir}...")
    parts = sorted(glob.glob(str(table_dir / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {table_dir}")

    if max_parts is not None:
        parts = parts[:max_parts]
        print(f"Using first {len(parts)} parquet parts (max_parts={max_parts})")

    dfs = []
    for i, part_path in enumerate(parts):
        if i % 50 == 0:
            print(f"  Loaded {i}/{len(parts)} parts...")
        part_df = pd.read_parquet(part_path)
        if sample_frac is not None and 0.0 < sample_frac < 1.0:
            part_df = part_df.sample(frac=sample_frac, random_state=42)
        dfs.append(part_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    return df


def compute_recall_at_k(df: pd.DataFrame, score_col: str, k: int = 20) -> float:
    """Compute mean recall@k over (session, target) groups."""
    recalls: List[float] = []

    for (_, _), group in df.groupby(["session", "target"], sort=False):
        group_sorted = group.sort_values(score_col, ascending=False)
        top_k = group_sorted.head(k)

        positives_in_top_k = int(top_k["label"].sum())
        total_positives = int(group_sorted["label"].sum())

        if total_positives == 0:
            continue

        recall = positives_in_top_k / min(k, total_positives)
        recalls.append(recall)

    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def build_model(hidden_dim: int, learning_rate_init: float, max_iter: int) -> Pipeline:
    """Create a simple standardized MLP pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=4096,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )


def train_target_model(
    df_target: pd.DataFrame,
    target: str,
    output_dir: Path,
    hidden_dim: int,
    learning_rate_init: float,
    max_iter: int,
) -> Tuple[Pipeline, Dict]:
    """Train and evaluate one MLP model for a single target."""
    print(f"\n{'=' * 60}")
    print(f"Training MLP for target: {target}")
    print(f"{'=' * 60}")

    train_df = df_target[df_target["split"] == "train"].copy()
    valid_df = df_target[df_target["split"] == "valid"].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"Target={target} has insufficient split rows: train={len(train_df)}, valid={len(valid_df)}"
        )

    print(f"Train set: {len(train_df):,} rows")
    print(f"Valid set: {len(valid_df):,} rows")

    X_train = train_df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y_train = train_df["label"].astype(int).values

    X_valid = valid_df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y_valid = valid_df["label"].astype(int).values

    print(
        f"Train label dist: positives={int(y_train.sum())}, negatives={int(len(y_train) - y_train.sum())}"
    )
    print(
        f"Valid label dist: positives={int(y_valid.sum())}, negatives={int(len(y_valid) - y_valid.sum())}"
    )

    model = build_model(
        hidden_dim=hidden_dim,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
    )

    print(
        "Training with params: "
        f"hidden_dim={hidden_dim}, lr={learning_rate_init}, max_iter={max_iter}"
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_valid = model.predict_proba(X_valid)[:, 1]

    ll_train = float(log_loss(y_train, y_pred_train, labels=[0, 1]))
    ll_valid = float(log_loss(y_valid, y_pred_valid, labels=[0, 1]))

    valid_eval = valid_df.copy()
    valid_eval["model_score"] = y_pred_valid

    recall_heuristic = compute_recall_at_k(valid_df, "heuristic_score", k=20)
    recall_mlp = compute_recall_at_k(valid_eval, "model_score", k=20)

    print(f"Train logloss: {ll_train:.5f}")
    print(f"Valid logloss: {ll_valid:.5f}")
    print(f"Recall@20 (Heuristic): {recall_heuristic:.5f}")
    print(f"Recall@20 (MLP):       {recall_mlp:.5f}")

    model_path = output_dir / f"mlp_model_{target}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    return model, {
        "target": target,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "train_logloss": ll_train,
        "valid_logloss": ll_valid,
        "recall_heuristic": float(recall_heuristic),
        "recall_mlp": float(recall_mlp),
        "improvement_pct": (
            float((recall_mlp - recall_heuristic) / recall_heuristic * 100.0)
            if recall_heuristic > 0
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple MLP models on OTTO candidate-level training table"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/training_table_full_clean_v3"),
        help="Directory with part-*.parquet training table files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mlp_models_simple"),
        help="Directory to save MLP models and metrics.json",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden size for the first MLP layer (second layer uses hidden_dim//2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=40,
        help="Maximum MLP iterations",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Optional cap for number of part-*.parquet files to read",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional per-part row sample fraction in (0,1)",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_table(
        args.data_dir,
        max_parts=args.max_parts,
        sample_frac=args.sample_frac,
    )

    all_metrics: List[Dict] = []
    for target in ["clicks", "carts", "orders"]:
        target_df = df[df["target"] == target].copy()
        if target_df.empty:
            print(f"Warning: no rows for target={target}, skipping")
            continue

        _, metrics = train_target_model(
            df_target=target_df,
            target=target,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            learning_rate_init=args.learning_rate,
            max_iter=args.max_iter,
        )
        all_metrics.append(metrics)

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")

    if len(all_metrics) == 3:
        weighted_heur = (
            0.10 * all_metrics[0]["recall_heuristic"]
            + 0.30 * all_metrics[1]["recall_heuristic"]
            + 0.60 * all_metrics[2]["recall_heuristic"]
        )
        weighted_mlp = (
            0.10 * all_metrics[0]["recall_mlp"]
            + 0.30 * all_metrics[1]["recall_mlp"]
            + 0.60 * all_metrics[2]["recall_mlp"]
        )
        print("\nWeighted Recall (0.10 clicks + 0.30 carts + 0.60 orders)")
        print(f"Heuristic: {weighted_heur:.5f}")
        print(f"MLP:       {weighted_mlp:.5f}")
        if weighted_heur > 0:
            print(f"Improvement: {(weighted_mlp - weighted_heur) / weighted_heur * 100.0:+.2f}%")


if __name__ == "__main__":
    main()
