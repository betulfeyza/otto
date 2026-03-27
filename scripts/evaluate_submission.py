#!/usr/bin/env python
"""
Evaluate a submission CSV against test labels without re-running the pipeline.

This is a lightweight evaluation script that only reads the submission CSV
and test labels, computes OTTO metrics (Recall@20), and reports results.

Usage:
    python scripts/evaluate_submission.py \
        --submission-path outputs/submission.csv \
        --labels-path archive/test_labels.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate submission CSV against test labels (fast, no pipeline re-run)"
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        required=True,
        help="Path to submission CSV file",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        required=True,
        help="Path to test_labels.parquet",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top items to consider (default: 20)",
    )

    args = parser.parse_args()

    # Import here to allow sys.path setup before package import
    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from otto_phase1.evaluate import evaluate_submission_file

    try:
        metrics = evaluate_submission_file(
            labels_path=args.labels_path,
            submission_csv_path=args.submission_path,
            topk=args.topk,
        )

        print("Evaluation Results (OTTO Metric):")
        print("-" * 50)
        for key in sorted(metrics.keys()):
            if key == "weighted_score":
                print(f"{key:.<40} {metrics[key]:.6f}")
            else:
                print(f"{key:.<40} {metrics[key]:.6f}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
