from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

EXPECTED_COLUMNS = ["session", "aid", "ts", "type"]


def list_parquet_files(parquet_dir: Path) -> List[Path]:
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    return files


def read_events_parquet(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path, columns=EXPECTED_COLUMNS)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")

    # Normalize and sort deterministically by session-time.
    df = df[EXPECTED_COLUMNS].copy()
    df["session"] = df["session"].astype("int64")
    df["aid"] = df["aid"].astype("int64")
    df["ts"] = df["ts"].astype("int64")
    df["type"] = df["type"].astype("string")
    return df.sort_values(["session", "ts"], kind="mergesort")


def iter_shard_dataframes(parquet_dir: Path) -> Iterator[pd.DataFrame]:
    for file_path in list_parquet_files(parquet_dir):
        yield read_events_parquet(file_path)


def iter_sorted_test_sessions(test_dir: Path) -> Iterable[pd.DataFrame]:
    # Returns per-shard, session-grouped sorted events for prediction phase.
    for shard_df in iter_shard_dataframes(test_dir):
        yield shard_df
