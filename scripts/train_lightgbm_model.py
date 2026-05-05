#!/usr/bin/env python3.14
"""
Train Gradient Boosting models on candidate-level training data.

Supports smoke/benchmark/full training tables.
Outputs: models, feature importance, validation metrics.
Uses sklearn's GradientBoostingClassifier (no external dependencies like libomp).
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error


# Feature columns used in training
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


def load_training_table(table_dir: Path) -> pd.DataFrame:
    """Load all parquet parts from training table directory."""
    print(f"Loading training table from {table_dir}...")
    parts = sorted(glob.glob(str(table_dir / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {table_dir}")
    
    dfs = []
    for i, part_path in enumerate(parts):
        if i % 50 == 0:
            print(f"  Loaded {i}/{len(parts)} parts...")
        dfs.append(pd.read_parquet(part_path))
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    return df


def train_target_model(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    hyperparams: Dict,
) -> Tuple[GradientBoostingClassifier, Dict]:
    """Train a single Gradient Boosting model for one target."""
    
    print(f"\n{'='*60}")
    print(f"Training Gradient Boosting for target: {target}")
    print(f"{'='*60}")
    
    # Split train/valid based on 'split' column
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    
    print(f"Train set: {len(train_df):,} rows")
    print(f"Valid set: {len(valid_df):,} rows")
    
    # Prepare X, y
    X_train = train_df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y_train = train_df["label"].astype(int).values
    
    X_valid = valid_df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y_valid = valid_df["label"].astype(int).values
    
    # Heuristic baseline scores
    heuristic_train = train_df["heuristic_score"].values
    heuristic_valid = valid_df["heuristic_score"].values
    
    print(f"Train label dist: clicks={np.sum(y_train)} positives, {len(y_train) - np.sum(y_train)} negatives")
    print(f"Valid label dist: clicks={np.sum(y_valid)} positives, {len(y_valid) - np.sum(y_valid)} negatives")
    
    # Train model with warm_start to monitor validation loss
    print(f"\nTraining with hyperparams: {hyperparams}")
    model = GradientBoostingClassifier(**hyperparams, random_state=42, warm_start=True)
    
    best_val_loss = float("inf")
    best_model = None
    patience = 50
    patience_counter = 0
    
    for iteration in range(1, 501):
        model.n_estimators = iteration
        model.fit(X_train, y_train)
        
        val_loss = -model.score(X_valid, y_valid)  # Negative for minimization
        
        if iteration % 50 == 0 or iteration == 1:
            train_loss = -model.score(X_train, y_train)
            print(f"  [Round {iteration:3d}] Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at round {iteration} (no improvement for {patience} rounds)")
                break
    
    model = best_model
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 10 features for {target}:")
    print(importance.head(10).to_string(index=False))
    
    # Predictions
    y_pred_train_proba = model.predict_proba(X_train)[:, 1]
    y_pred_valid_proba = model.predict_proba(X_valid)[:, 1]
    
    # Add predictions to valid_df for recall@20 calculation
    valid_df_eval = valid_df.copy()
    valid_df_eval["model_score"] = y_pred_valid_proba
    
    # MSE
    mse_train = mean_squared_error(y_train, y_pred_train_proba)
    mse_valid = mean_squared_error(y_valid, y_pred_valid_proba)
    
    print(f"\nTrain MSE: {mse_train:.4f}")
    print(f"Valid MSE: {mse_valid:.4f}")
    
    # Recall@20 comparison (heuristic vs Gradient Boosting)
    recall_heuristic = compute_recall_at_k(
        valid_df, "heuristic_score", k=20
    )
    recall_gb = compute_recall_at_k(
        valid_df_eval, "model_score", k=20
    )
    
    print(f"\nRecall@20 (Heuristic):          {recall_heuristic:.4f}")
    print(f"Recall@20 (Gradient Boosting): {recall_gb:.4f}")
    if recall_heuristic > 0:
        improvement = (recall_gb - recall_heuristic) / recall_heuristic * 100
        print(f"Improvement:                   {improvement:+.2f}%")
    
    # Save model
    model_path = output_dir / f"gb_model_{target}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    # Prepare metrics dict
    metrics = {
        "target": target,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "train_mse": float(mse_train),
        "valid_mse": float(mse_valid),
        "recall_heuristic": float(recall_heuristic),
        "recall_gb": float(recall_gb),
        "improvement_pct": float((recall_gb - recall_heuristic) / recall_heuristic * 100) if recall_heuristic > 0 else 0.0,
        "top_5_features": importance.head(5)[["feature", "importance"]].to_dict("records"),
    }
    
    return model, metrics


def compute_recall_at_k(
    df: pd.DataFrame,
    score_col: str,
    k: int = 20,
) -> float:
    """
    Compute recall@k CORRECTLY (per session-target).
    
    For each (session, target) pair:
      - Rank candidates by score
      - Check if positives in top-k
      - Calculate recall = |top-k ∩ positives| / min(k, |positives|)
    
    Then average across all session-target pairs.
    """
    recall_list = []
    
    for (session, target), group in df.groupby(["session", "target"]):
        # Group has candidates for this session-target
        group = group.copy().sort_values(score_col, ascending=False)
        
        # Top-k candidates
        top_k = group.head(k)
        
        # Count positives in top-k
        positives_in_top_k = top_k["label"].sum()
        total_positives = group["label"].sum()
        
        if total_positives == 0:
            # No positives for this session-target, skip
            continue
        
        recall = positives_in_top_k / min(k, total_positives)
        recall_list.append(recall)
    
    if not recall_list:
        return 0.0
    
    return np.mean(recall_list)


def main():
    parser = argparse.ArgumentParser(description="Train Gradient Boosting models on training table")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/training_table_smoke"),
        help="Training table directory (smoke/benchmark/full)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gb_models"),
        help="Output directory for models and metrics",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Max depth of trees",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (eta)",
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_training_table(args.data_dir)
    
    # Sklearn GradientBoosting hyperparams
    hyperparams = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "max_features": 0.8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    }
    
    # Train models for each target
    all_metrics = []
    for target in ["clicks", "carts", "orders"]:
        # Filter data for this target
        target_df = df[df["target"] == target].copy()
        if len(target_df) == 0:
            print(f"Warning: No data for target {target}")
            continue
        
        print(f"\nTarget {target}: {len(target_df):,} rows")
        
        model, metrics = train_target_model(
            target_df,
            target,
            output_dir,
            hyperparams,
        )
        all_metrics.append(metrics)
    
    # Summary report
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - SMOKE TEST RESULTS")
    print(f"{'='*60}")
    for m in all_metrics:
        print(f"\n{m['target'].upper()}:")
        print(f"  Recall@20 Heuristic:          {m['recall_heuristic']:.4f}")
        print(f"  Recall@20 Gradient Boosting:  {m['recall_gb']:.4f}")
        print(f"  Improvement:                  {m['improvement_pct']:+.2f}%")
    
    weighted_heur = (
        0.10 * all_metrics[0]["recall_heuristic"] +
        0.30 * all_metrics[1]["recall_heuristic"] +
        0.60 * all_metrics[2]["recall_heuristic"]
    )
    weighted_gb = (
        0.10 * all_metrics[0]["recall_gb"] +
        0.30 * all_metrics[1]["recall_gb"] +
        0.60 * all_metrics[2]["recall_gb"]
    )
    
    print(f"\nWeighted Recall (0.10 clicks + 0.30 carts + 0.60 orders):")
    print(f"  Heuristic Baseline:   {weighted_heur:.4f}")
    print(f"  Gradient Boosting:    {weighted_gb:.4f}")
    print(f"  Improvement:          {(weighted_gb - weighted_heur) / weighted_heur * 100:+.2f}%")
    print(f"{'='*60}")
    print(f"\nNote: This is smoke test (1 shard) - expect ±0.02 variance on full data")


if __name__ == "__main__":
    main()
