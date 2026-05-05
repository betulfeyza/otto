#!/usr/bin/env python3.14
"""Train LightGBM models on candidate-level training data."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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

BASE_COLS = ["split", "session", "target", "label", "heuristic_score"]
LOAD_COLS = BASE_COLS + FEATURE_COLS


def load_training_table_for_target(
    table_dir: Path,
    target: str,
    max_parts: int | None = None,
) -> pd.DataFrame:
    print(f"Loading training table from {table_dir} for target={target}...")
    parts = sorted(glob.glob(str(table_dir / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {table_dir}")

    if max_parts:
        print(f"  Limiting to {max_parts} parts (out of {len(parts)} available)")
        parts = parts[:max_parts]

    dfs = []
    for i, part_path in enumerate(parts):
        if i % 20 == 0:
            print(f"  Loaded {i}/{len(parts)} parts...")
        part_df = pd.read_parquet(part_path, columns=LOAD_COLS)
        part_df = part_df[part_df["target"] == target]
        if not part_df.empty:
            dfs.append(part_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    return df


def compute_recall_at_k(df: pd.DataFrame, score_col: str, k: int = 20) -> float:
    recall_list = []
    for (_, _), group in df.groupby(["session", "target"], sort=False):
        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)

        positives_in_top_k = float(top_k["label"].sum())
        total_positives = float(group["label"].sum())
        if total_positives == 0:
            continue
        recall_list.append(positives_in_top_k / min(float(k), total_positives))

    if not recall_list:
        return 0.0
    return float(np.mean(recall_list))


def train_target_model(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Training LightGBM for target: {target}")
    print(f"{'='*60}")

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()

    X_train = train_df[FEATURE_COLS].fillna(0.0)
    y_train = train_df["label"].astype(int)
    X_valid = valid_df[FEATURE_COLS].fillna(0.0)
    y_valid = valid_df["label"].astype(int)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print(f"Train rows: {len(train_df):,}, Valid rows: {len(valid_df):,}")
    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    y_pred_train = booster.predict(X_train, num_iteration=booster.best_iteration)
    y_pred_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)

    valid_eval = valid_df.copy()
    valid_eval["model_score"] = y_pred_valid

    mse_train = float(mean_squared_error(y_train, y_pred_train))
    mse_valid = float(mean_squared_error(y_valid, y_pred_valid))
    mse_gap = mse_valid - mse_train
    recall_heuristic = compute_recall_at_k(valid_df, "heuristic_score", k=20)
    recall_lgb = compute_recall_at_k(valid_eval, "model_score", k=20)

    importance_gain = booster.feature_importance(importance_type="gain")
    imp = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": importance_gain})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    print("Top features:\n" + imp.to_string(index=False))

    model_path = output_dir / f"lgb_model_{target}.txt"
    booster.save_model(str(model_path))

    print(f"Saved: {model_path}")
    print(f"Recall@20 Heuristic: {recall_heuristic:.4f}")
    print(f"Recall@20 LightGBM:  {recall_lgb:.4f}")

    return {
        "target": target,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "best_iteration": int(booster.best_iteration),
        "train_mse": mse_train,
        "valid_mse": mse_valid,
        "mse_gap_valid_minus_train": float(mse_gap),
        "recall_heuristic": float(recall_heuristic),
        "recall_lgb": float(recall_lgb),
        "improvement_pct": float(((recall_lgb - recall_heuristic) / recall_heuristic * 100.0) if recall_heuristic > 0 else 0.0),
        "top_5_features": imp.head(5).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM on candidate-level table")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/training_table_benchmark_1shard"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lgb_models_benchmark_v1"))
    parser.add_argument("--max-parts", type=int, default=None, help="Limit number of parquet parts to load (for memory constraints)")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-l2", type=float, default=10.0)
    parser.add_argument("--min-gain-to-split", type=float, default=0.05)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_gain_to_split": args.min_gain_to_split,
        "seed": 42,
        "verbosity": -1,
    }

    all_metrics = []
    for target in ["clicks", "carts", "orders"]:
        target_df = load_training_table_for_target(
            args.data_dir,
            target=target,
            max_parts=args.max_parts,
        )
        if target_df.empty:
            continue
        all_metrics.append(
            train_target_model(
                target_df,
                target,
                args.output_dir,
                params,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
            )
        )

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    if len(all_metrics) == 3:
        weighted_heur = 0.10 * all_metrics[0]["recall_heuristic"] + 0.30 * all_metrics[1]["recall_heuristic"] + 0.60 * all_metrics[2]["recall_heuristic"]
        weighted_lgb = 0.10 * all_metrics[0]["recall_lgb"] + 0.30 * all_metrics[1]["recall_lgb"] + 0.60 * all_metrics[2]["recall_lgb"]
        print("\nWeighted Recall:")
        print(f"Heuristic: {weighted_heur:.4f}")
        print(f"LightGBM:  {weighted_lgb:.4f}")
        print(f"Improvement: {((weighted_lgb - weighted_heur) / weighted_heur * 100.0):+.2f}%")

    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
