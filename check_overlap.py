#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from glob import glob

data_root = Path('archive')

print('=' * 60)
print('CHECKING TEST vs TRAIN SESSION OVERLAP')
print('=' * 60)

# Load test sessions (first 5 parts)
print('\n[1/4] Loading test sessions...')
test_sessions = set()
test_ts_all = []
for part in sorted(glob(str(data_root / 'test_parquet' / 'part-*.parquet')))[:5]:
    df = pd.read_parquet(part)
    test_sessions.update(df['session'].unique())
    test_ts_all.extend(df['ts'].tolist())
    print(f'  {Path(part).name}: {len(df)} events, {df["session"].nunique()} sessions')

print(f'Total test (sampled): {len(test_sessions)} unique sessions')

# Load train sessions (first 5 parts)
print('\n[2/4] Loading train sessions...')
train_sessions = set()
train_ts_all = []
for part in sorted(glob(str(data_root / 'train_parquet' / 'part-*.parquet')))[:5]:
    df = pd.read_parquet(part)
    train_sessions.update(df['session'].unique())
    train_ts_all.extend(df['ts'].tolist())
    print(f'  {Path(part).name}: {len(df)} events, {df["session"].nunique()} sessions')

print(f'Total train (sampled): {len(train_sessions)} unique sessions')

# Session overlap
print('\n[3/4] Session Overlap Analysis:')
overlap = test_sessions & train_sessions
overlap_pct = 100 * len(overlap) / len(test_sessions) if test_sessions else 0
print(f'  Overlap: {len(overlap)} / {len(test_sessions)} ({overlap_pct:.1f}%)')

# Temporal analysis
print('\n[4/4] Temporal Analysis:')
test_min = min(test_ts_all) if test_ts_all else 0
test_max = max(test_ts_all) if test_ts_all else 0
train_min = min(train_ts_all) if train_ts_all else 0
train_max = max(train_ts_all) if train_ts_all else 0

print(f'  Train TS: {int(train_min)} → {int(train_max)}')
print(f'  Test TS:  {int(test_min)} → {int(test_max)}')

if test_min <= train_max and train_min <= test_max:
    print(f'  ⚠️ TEMPORAL OVERLAP: {int(max(test_min, train_min))} → {int(min(test_max, train_max))}')
else:
    print(f'  ✅ TEMPORAL SEPARATION: Test events after train!')

print('=' * 60)
