#!/usr/bin/env python3
"""
Memory-safe full training for simple MLP models on OTTO candidate-level data.

This script avoids loading all parquet parts into RAM at once:
1) Fit StandardScaler incrementally on train rows.
2) Train MLPClassifier with partial_fit over parquet parts for multiple epochs.
3) Evaluate logloss and sampled Recall@20 on validation rows.

Outputs:
- mlp_model_clicks.pkl
- mlp_model_carts.pkl
- mlp_model_orders.pkl
- metrics.json
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
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

META_COLS = ["session", "target", "label", "heuristic_score", "split"]
ALL_COLS = list(dict.fromkeys(META_COLS + FEATURE_COLS))


def get_parts(data_dir: Path, max_parts: int | None) -> List[str]:
    parts = sorted(glob.glob(str(data_dir / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    if max_parts is not None:
        parts = parts[:max_parts]
    return parts


def iter_target_split_batches(
    parts: List[str],
    target: str,
    split: str,
    sample_frac: float | None,
) -> Iterator[pd.DataFrame]:
    for i, part in enumerate(parts):
        if i % 50 == 0:
            print(f"  reading {split}/{target}: {i}/{len(parts)}")

        df = pd.read_parquet(part, columns=ALL_COLS)
        df = df[(df["target"] == target) & (df["split"] == split)]
        if df.empty:
            continue

        if sample_frac is not None and 0.0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        if df.empty:
            continue

        yield df


def compute_recall_at_k(df: pd.DataFrame, score_col: str, k: int = 20) -> float:
    recalls: List[float] = []
    for (_, _), group in df.groupby(["session", "target"], sort=False):
        g = group.sort_values(score_col, ascending=False)
        positives_total = int(g["label"].sum())
        if positives_total == 0:
            continue
        positives_top_k = int(g.head(k)["label"].sum())
        recalls.append(positives_top_k / min(k, positives_total))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def binary_logloss_sum(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_scaler(parts: List[str], target: str, sample_frac: float | None) -> Dict[str, int]:
    scaler = StandardScaler()
    total_rows = 0
    total_pos = 0

    print(f"\n[1/3] Fitting scaler for target={target}")
    for batch_df in iter_target_split_batches(parts, target, "train", sample_frac):
        X = batch_df[FEATURE_COLS].fillna(0.0).astype(np.float32).values
        y = batch_df["label"].astype(np.int8).values
        scaler.partial_fit(X)
        total_rows += len(batch_df)
        total_pos += int(y.sum())

    return {"rows": total_rows, "pos": total_pos, "scaler": scaler}


def train_mlp_stream(
    parts: List[str],
    scaler: StandardScaler,
    target: str,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    sample_frac: float | None,
) -> MLPClassifier:
    print(f"\n[2/3] Training MLP for target={target}")
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=42,
        early_stopping=False,
        verbose=False,
    )

    is_first = True
    for epoch in range(1, epochs + 1):
        epoch_rows = 0
        epoch_pos = 0
        print(f"  epoch {epoch}/{epochs}")

        for batch_df in iter_target_split_batches(parts, target, "train", sample_frac):
            X = batch_df[FEATURE_COLS].fillna(0.0).astype(np.float32).values
            y = batch_df["label"].astype(np.int8).values
            X_scaled = scaler.transform(X)

            if is_first:
                clf.partial_fit(X_scaled, y, classes=np.array([0, 1], dtype=np.int8))
                is_first = False
            else:
                clf.partial_fit(X_scaled, y)

            epoch_rows += len(batch_df)
            epoch_pos += int(y.sum())

        print(f"    rows={epoch_rows:,}, positives={epoch_pos:,}")

    return clf


def evaluate_stream(
    parts: List[str],
    target: str,
    scaler: StandardScaler,
    clf: MLPClassifier,
    sample_frac: float | None,
    recall_eval_cap_rows: int,
) -> Dict[str, float]:
    print(f"\n[3/3] Evaluating target={target}")

    ll_sum = 0.0
    row_count = 0
    pos_count = 0
    recall_frames: List[pd.DataFrame] = []
    recall_rows = 0

    for batch_df in iter_target_split_batches(parts, target, "valid", sample_frac):
        X = batch_df[FEATURE_COLS].fillna(0.0).astype(np.float32).values
        y = batch_df["label"].astype(np.int8).values

        p = clf.predict_proba(scaler.transform(X))[:, 1]

        ll_sum += binary_logloss_sum(y.astype(np.float64), p.astype(np.float64))
        row_count += len(batch_df)
        pos_count += int(y.sum())

        if recall_rows < recall_eval_cap_rows:
            keep = min(len(batch_df), recall_eval_cap_rows - recall_rows)
            frame = batch_df.iloc[:keep][["session", "target", "label", "heuristic_score"]].copy()
            frame["model_score"] = p[:keep]
            recall_frames.append(frame)
            recall_rows += keep

    if row_count == 0:
        raise ValueError(f"No validation rows found for target={target}")

    valid_logloss = ll_sum / row_count

    recall_heuristic = None
    recall_mlp = None
    if recall_frames:
        recall_df = pd.concat(recall_frames, ignore_index=True)
        recall_heuristic = compute_recall_at_k(recall_df, "heuristic_score", k=20)
        recall_mlp = compute_recall_at_k(recall_df, "model_score", k=20)

    return {
        "valid_size": int(row_count),
        "valid_positives": int(pos_count),
        "valid_logloss": float(valid_logloss),
        "recall_heuristic": float(recall_heuristic) if recall_heuristic is not None else None,
        "recall_mlp": float(recall_mlp) if recall_mlp is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming full-data MLP training for OTTO candidate table"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/training_table_full_clean_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mlp_models_streaming_v1"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument(
        "--recall-eval-cap-rows",
        type=int,
        default=2_000_000,
        help="Cap validation rows kept in RAM for recall@20 computation",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parts = get_parts(args.data_dir, args.max_parts)
    print(f"Using {len(parts)} parquet parts from {args.data_dir}")

    all_metrics: List[Dict] = []
    for target in ["clicks", "carts", "orders"]:
        print(f"\n{'=' * 70}\nTarget: {target}\n{'=' * 70}")

        scaler_info = fit_scaler(parts, target, args.sample_frac)
        train_rows = scaler_info["rows"]
        train_pos = scaler_info["pos"]
        scaler = scaler_info["scaler"]

        if train_rows == 0:
            print(f"No training rows for {target}, skipping")
            continue

        train_neg = train_rows - train_pos
        print(f"Train stats: rows={train_rows:,}, pos={train_pos:,}, neg={train_neg:,}")

        clf = train_mlp_stream(
            parts=parts,
            scaler=scaler,
            target=target,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            sample_frac=args.sample_frac,
        )

        eval_metrics = evaluate_stream(
            parts=parts,
            target=target,
            scaler=scaler,
            clf=clf,
            sample_frac=args.sample_frac,
            recall_eval_cap_rows=args.recall_eval_cap_rows,
        )

        model = Pipeline(steps=[("scaler", scaler), ("mlp", clf)])
        model_path = args.output_dir / f"mlp_model_{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metrics = {
            "target": target,
            "train_size": int(train_rows),
            "train_positives": int(train_pos),
            "valid_size": eval_metrics["valid_size"],
            "valid_positives": eval_metrics["valid_positives"],
            "valid_logloss": eval_metrics["valid_logloss"],
            "recall_heuristic": eval_metrics["recall_heuristic"],
            "recall_mlp": eval_metrics["recall_mlp"],
            "improvement_pct": (
                float(
                    (eval_metrics["recall_mlp"] - eval_metrics["recall_heuristic"])
                    / eval_metrics["recall_heuristic"]
                    * 100.0
                )
                if eval_metrics["recall_heuristic"] not in (None, 0.0)
                and eval_metrics["recall_mlp"] is not None
                else None
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sample_frac": args.sample_frac,
            "max_parts": args.max_parts,
            "recall_eval_cap_rows": args.recall_eval_cap_rows,
        }
        all_metrics.append(metrics)

        print(f"Saved model: {model_path}")
        print(
            f"Validation logloss={metrics['valid_logloss']:.6f}, "
            f"recall@20 heuristic={metrics['recall_heuristic']}, "
            f"recall@20 mlp={metrics['recall_mlp']}"
        )

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
