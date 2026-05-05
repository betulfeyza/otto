#!/usr/bin/env python3
"""Duplicate valid-only aids into train split for an existing training table.

This script works on a parquet-part training table directory. It finds aids
that appear in the valid split but not in the train split, then:

- copies every matching valid row into the train split
- optionally removes those aids from the original valid rows

The resulting table can be used to train models where every aid that appears
in validation is also present in training.
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path
from typing import Iterable, Set, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate valid-only aids into train for a training table"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/training_table_full_clean_v2"),
        help="Directory containing part-*.parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_table_full_clean_v2_valid_dup_train"),
        help="Directory where the transformed parquet parts will be written",
    )
    parser.add_argument(
        "--keep-valid-only-in-valid",
        action="store_true",
        help="Keep the original valid rows as well as the train duplicates",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=0,
        help="If >0, only process the first N parquet parts (smoke test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory",
    )
    return parser.parse_args()


def _part_paths(source_dir: Path, max_parts: int) -> list[Path]:
    parts = [Path(p) for p in sorted(glob.glob(str(source_dir / "part-*.parquet")))]
    if max_parts and max_parts > 0:
        parts = parts[: int(max_parts)]
    return parts


def _collect_aid_sets(parts: Iterable[Path]) -> Tuple[Set[int], Set[int]]:
    train_aids: Set[int] = set()
    valid_aids: Set[int] = set()

    for i, part_path in enumerate(parts, start=1):
        part_df = pd.read_parquet(part_path, columns=["split", "aid"])
        if "split" not in part_df.columns or "aid" not in part_df.columns:
            raise ValueError(f"Missing required columns in {part_path}")

        train_mask = part_df["split"] == "train"
        valid_mask = part_df["split"] == "valid"

        if train_mask.any():
            train_aids.update(int(aid) for aid in part_df.loc[train_mask, "aid"].unique())
        if valid_mask.any():
            valid_aids.update(int(aid) for aid in part_df.loc[valid_mask, "aid"].unique())

        if i % 50 == 0:
            print(f"  scanned {i} parts...")

    return train_aids, valid_aids


def _write_part(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    parts = _part_paths(source_dir, args.max_parts)
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in {source_dir}")

    if output_dir.exists():
        existing_parts = list(output_dir.glob("part-*.parquet"))
        if existing_parts and not args.overwrite:
            raise FileExistsError(
                f"Output directory already contains parquet parts: {output_dir}. "
                "Use --overwrite to replace it."
            )
        if args.overwrite:
            shutil.rmtree(output_dir)

    print(f"Source dir: {source_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Parts to process: {len(parts)}")
    print("[1/3] Collecting train/valid aid sets...")
    train_aids, valid_aids = _collect_aid_sets(parts)
    valid_only_aids = valid_aids - train_aids

    print(f"  unique train aids: {len(train_aids):,}")
    print(f"  unique valid aids: {len(valid_aids):,}")
    print(f"  valid-only aids:   {len(valid_only_aids):,}")

    if not valid_only_aids:
        print("No valid-only aids found. Copying source parts unchanged.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for part_path in parts:
            shutil.copy2(part_path, output_dir / part_path.name)
        return

    print("[2/3] Rewriting parquet parts...")
    rows_copied_to_train = 0
    rows_removed_from_valid = 0

    for i, part_path in enumerate(parts, start=1):
        df = pd.read_parquet(part_path)
        if "split" not in df.columns or "aid" not in df.columns:
            raise ValueError(f"Missing required columns in {part_path}")

        valid_only_mask = df["aid"].isin(valid_only_aids)
        valid_rows_mask = df["split"] == "valid"

        train_copy = df.loc[valid_only_mask].copy()
        if not train_copy.empty:
            train_copy.loc[:, "split"] = "train"
            rows_copied_to_train += int(len(train_copy))

        if args.keep_valid_only_in_valid:
            transformed = pd.concat([df, train_copy], ignore_index=True)
        else:
            valid_drop_mask = valid_rows_mask & valid_only_mask
            rows_removed_from_valid += int(valid_drop_mask.sum())
            kept = df.loc[~valid_drop_mask].copy()
            transformed = pd.concat([kept, train_copy], ignore_index=True)

        _write_part(transformed, output_dir / part_path.name)

        if i % 50 == 0:
            print(f"  wrote {i} parts...")

    print("[3/3] Done.")
    print(f"  copied rows to train:   {rows_copied_to_train:,}")
    if args.keep_valid_only_in_valid:
        print("  kept original valid rows: yes")
    else:
        print(f"  removed valid rows:      {rows_removed_from_valid:,}")
    print(f"  output written to:       {output_dir}")


if __name__ == "__main__":
    main()