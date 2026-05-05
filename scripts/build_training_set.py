from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-level training table with label join"
    )
    parser.add_argument("--data-root", type=Path, default=Path("archive"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_table"),
        help="Directory where parquet parts will be written",
    )

    parser.add_argument(
        "--holdout-days",
        type=int,
        default=7,
        help="Validation window from end of timeline in days",
    )
    parser.add_argument(
        "--train-prefix-ratio",
        type=float,
        default=0.8,
        help="Session-local train split ratio for sessions before cutoff",
    )
    parser.add_argument(
        "--max-candidates-per-target",
        type=int,
        default=200,
        help="Maximum candidate rows kept per session-target before sampling",
    )
    parser.add_argument(
        "--max-negatives-per-target",
        type=int,
        default=100,
        help="Maximum negatives kept per session-target",
    )
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=600000,
        help="Max rows per parquet part file",
    )
    parser.add_argument(
        "--gt-injection-mode",
        type=str,
        default="train",
        choices=("none", "train", "both"),
        help="Inject future GT aids into candidate set: none, train, or both",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If >0, only first N train shards are processed (debug mode)",
    )

    return parser.parse_args()


def main() -> None:
    from otto_phase1.config import BaselineConfig
    from otto_phase1.covisitation import build_covisitation_from_train
    from otto_phase1.io_utils import iter_shard_dataframes
    from otto_phase1.popularity import build_popularity_from_train
    from otto_phase1.training_data import (
        build_candidate_level_training_table,
        infer_global_cutoff_ts_from_max_ts,
    )

    args = parse_args()

    config = BaselineConfig(data_root=args.data_root)

    def maybe_limit_shards(shards_iter):
        if args.max_shards and args.max_shards > 0:
            return itertools.islice(shards_iter, int(args.max_shards))
        return shards_iter

    print("[1/4] Inferring time-based validation cutoff...")
    cutoff_ts = infer_global_cutoff_ts_from_max_ts(
        train_shards=maybe_limit_shards(iter_shard_dataframes(config.train_path)),
        holdout_days=args.holdout_days,
    )
    print(f"       cutoff_ts={cutoff_ts}")

    print("[2/4] Building popularity model from pre-cutoff train events...")
    popularity = build_popularity_from_train(
        train_shards=maybe_limit_shards(iter_shard_dataframes(config.train_path)),
        config=config,
        max_ts=cutoff_ts,
        cutoff_ts=cutoff_ts,
        train_prefix_ratio=args.train_prefix_ratio,
    )

    print("[3/4] Building co-visitation model from pre-cutoff train events...")
    covis = build_covisitation_from_train(
        train_shards=maybe_limit_shards(iter_shard_dataframes(config.train_path)),
        config=config,
        max_ts=cutoff_ts,
        cutoff_ts=cutoff_ts,
        train_prefix_ratio=args.train_prefix_ratio,
    )

    print("[4/4] Building candidate-level training table...")
    stats = build_candidate_level_training_table(
        train_shards=maybe_limit_shards(iter_shard_dataframes(config.train_path)),
        popularity=popularity,
        covisitation=covis,
        config=config,
        output_dir=args.output_dir,
        cutoff_ts=cutoff_ts,
        train_prefix_ratio=args.train_prefix_ratio,
        max_candidates_per_target=args.max_candidates_per_target,
        max_negatives_per_target=args.max_negatives_per_target,
        rows_per_part=args.rows_per_part,
        gt_injection_mode=args.gt_injection_mode,
    )

    print("Done.")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
