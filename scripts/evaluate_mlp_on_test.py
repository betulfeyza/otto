#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TARGETS = ["clicks", "carts", "orders"]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sklearn MLP models on archive/test_parquet with test_labels"
    )
    parser.add_argument("--data-root", type=Path, default=Path("archive"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("outputs/mlp_models_streaming_v1_full")
    )
    parser.add_argument(
        "--submission-out",
        type=Path,
        default=Path("outputs/submission_mlp_streaming_v1_full.csv"),
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=0,
        help="If >0, only first N train shards are used for popularity/covis",
    )
    parser.add_argument(
        "--max-test-shards",
        type=int,
        default=0,
        help="If >0, only first N test shards are predicted (debug mode)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache"),
        help="Directory used to load/save popularity and covis caches",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable reading/writing popularity and covis caches",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/eval_checkpoints_mlp"),
        help="Directory for per-shard prediction checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-shard prediction checkpoints",
    )
    return parser.parse_args()


def maybe_limit_shards(shards_iter: Iterable[pd.DataFrame], max_shards: int):
    if max_shards and max_shards > 0:
        return itertools.islice(shards_iter, int(max_shards))
    return shards_iter


def _load_models(model_dir: Path) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for target in TARGETS:
        model_path = model_dir / f"mlp_model_{target}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        with model_path.open("rb") as f:
            models[target] = pickle.load(f)
    return models


def _cache_key(max_train_shards: int) -> str:
    if max_train_shards and max_train_shards > 0:
        return f"train{int(max_train_shards)}"
    return "train_all"


def _cache_paths(cache_dir: Path, max_train_shards: int) -> Tuple[Path, Path]:
    key = _cache_key(max_train_shards)
    return (
        cache_dir / f"popularity_{key}.pkl",
        cache_dir / f"covis_{key}.pkl",
    )


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _shard_checkpoint_path(checkpoint_dir: Path, shard_idx: int) -> Path:
    return checkpoint_dir / f"predictions_shard_{int(shard_idx):03d}.pkl"


def _predict_target_topk(
    aids: List[int],
    feature_map: Dict[int, Dict[str, float]],
    model,
    topk: int,
) -> List[int]:
    if not aids:
        return []

    x = np.array(
        [
            [float(feature_map.get(int(aid), {}).get(col, 0.0)) for col in FEATURE_COLS]
            for aid in aids
        ],
        dtype=np.float32,
    )
    scores = model.predict_proba(x)[:, 1]

    ranked = sorted(
        zip(aids, scores),
        key=lambda t: (float(t[1]), -int(t[0])),
        reverse=True,
    )
    return [int(aid) for aid, _ in ranked[:topk]]


def main() -> None:
    t0 = time.time()
    args = parse_args()

    from otto_phase1.candidates import build_candidate_pool
    from otto_phase1.config import BaselineConfig
    from otto_phase1.covisitation import build_covisitation_from_train
    from otto_phase1.evaluate import evaluate_predictions
    from otto_phase1.io_utils import iter_shard_dataframes
    from otto_phase1.popularity import build_popularity_from_train
    from otto_phase1.predict import predictions_to_submission_df
    from otto_phase1.scoring import compute_candidate_feature_and_score_maps

    config = BaselineConfig(data_root=args.data_root)

    print("[1/5] Loading MLP models...")
    models = _load_models(args.model_dir)

    pop_cache_path, covis_cache_path = _cache_paths(args.cache_dir, args.max_train_shards)

    popularity = None
    covis = None
    if not args.disable_cache:
        if pop_cache_path.exists():
            print(f"[2/5] Loading popularity cache: {pop_cache_path}")
            popularity = _load_pickle(pop_cache_path)
        if covis_cache_path.exists():
            print(f"[3/5] Loading co-visitation cache: {covis_cache_path}")
            covis = _load_pickle(covis_cache_path)

    if popularity is None:
        print("[2/5] Building popularity from train shards...")
        popularity = build_popularity_from_train(
            train_shards=maybe_limit_shards(
                iter_shard_dataframes(config.train_path), args.max_train_shards
            ),
            config=config,
        )
        if not args.disable_cache:
            _save_pickle(pop_cache_path, popularity)
            print(f"       Saved popularity cache: {pop_cache_path}")

    if covis is None:
        print("[3/5] Building co-visitation from train shards...")
        covis = build_covisitation_from_train(
            train_shards=maybe_limit_shards(
                iter_shard_dataframes(config.train_path), args.max_train_shards
            ),
            config=config,
        )
        if not args.disable_cache:
            _save_pickle(covis_cache_path, covis)
            print(f"       Saved co-visitation cache: {covis_cache_path}")

    print("[4/5] Predicting test shards with MLP...")
    predictions: Dict[Tuple[int, str], List[int]] = {}
    test_iter = maybe_limit_shards(iter_shard_dataframes(config.test_path), args.max_test_shards)

    if args.resume:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx, shard_df in enumerate(test_iter, start=1):
        shard_t0 = time.time()
        shard_ckpt = _shard_checkpoint_path(args.checkpoint_dir, shard_idx)
        if args.resume and shard_ckpt.exists():
            shard_predictions = _load_pickle(shard_ckpt)
            predictions.update(shard_predictions)
            print(
                f"  shard {shard_idx}: reused checkpoint "
                f"({len(shard_predictions):,} session-target rows)"
            )
            continue

        print(f"  shard {shard_idx}: rows={len(shard_df):,}")
        shard_predictions: Dict[Tuple[int, str], List[int]] = {}
        for session, session_df in shard_df.groupby("session", sort=False):
            session = int(session)
            session_events = list(
                zip(
                    session_df["aid"].astype("int64").tolist(),
                    session_df["ts"].astype("int64").tolist(),
                    session_df["type"].astype("string").tolist(),
                )
            )

            shared_pool = build_candidate_pool(
                session_events=session_events,
                popularity=popularity,
                covisitation=covis,
                config=config,
            )

            aids = [int(aid) for aid in shared_pool.aids]
            for target in TARGETS:
                feature_map, _ = compute_candidate_feature_and_score_maps(
                    session_events=session_events,
                    candidate_pool=shared_pool,
                    popularity=popularity,
                    config=config,
                    target=target,
                )
                shard_predictions[(session, target)] = _predict_target_topk(
                    aids=aids,
                    feature_map=feature_map,
                    model=models[target],
                    topk=config.topk_per_target,
                )

        predictions.update(shard_predictions)
        if args.resume:
            _save_pickle(shard_ckpt, shard_predictions)

        elapsed_shard = time.time() - shard_t0
        print(
            f"    done shard {shard_idx} in {elapsed_shard:.1f}s, "
            f"session-target rows={len(shard_predictions):,}"
        )

    print("[5/5] Evaluating against test_labels.parquet...")
    if args.max_test_shards and args.max_test_shards > 0:
        labels_df = pd.read_parquet(config.labels_path)
        predicted_sessions = {session for (session, _) in predictions.keys()}
        labels_df = labels_df[labels_df["session"].astype("int64").isin(predicted_sessions)]

        tmp_labels_path = args.submission_out.parent / "_tmp_labels_subset.parquet"
        tmp_labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(tmp_labels_path, index=False)
        metrics = evaluate_predictions(
            labels_path=tmp_labels_path,
            predictions=predictions,
            config=config,
        )
        tmp_labels_path.unlink(missing_ok=True)
    else:
        metrics = evaluate_predictions(
            labels_path=config.labels_path,
            predictions=predictions,
            config=config,
        )

    submission_df = predictions_to_submission_df(predictions)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.submission_out, index=False)

    print("\nMetrics:")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.6f}")
    print(f"\nSubmission saved: {args.submission_out}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
