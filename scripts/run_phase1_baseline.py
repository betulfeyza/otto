from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OTTO Phase-1 heuristic baseline")
    parser.add_argument("--data-root", type=Path, default=Path("archive"))
    parser.add_argument("--save-submission", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--no-eval", action="store_true")

    # Tunable knobs
    parser.add_argument("--session-max-events-covis", type=int, default=30)
    parser.add_argument("--covis-topk-neighbors", type=int, default=40)
    parser.add_argument("--covis-candidates-per-item", type=int, default=25)
    parser.add_argument("--min-candidates-before-fallback", type=int, default=120)

    return parser.parse_args()


def main() -> None:
    from otto_phase1.config import BaselineConfig
    from otto_phase1.pipeline import run_phase1_pipeline

    args = parse_args()

    config = BaselineConfig(
        data_root=args.data_root,
        session_max_events_for_covis=args.session_max_events_covis,
        covis_topk_neighbors=args.covis_topk_neighbors,
        covis_candidates_per_item=args.covis_candidates_per_item,
        min_candidates_before_fallback=args.min_candidates_before_fallback,
    )

    metrics = run_phase1_pipeline(
        config=config,
        save_submission_to=args.save_submission,
        run_evaluation=not args.no_eval,
    )

    if metrics:
        print("Offline metrics:")
        for k in sorted(metrics.keys()):
            print(f"  {k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()
