import glob
import pandas as pd

parts = sorted(glob.glob('outputs/training_table_benchmark_1shard/part-*.parquet'))
print('Benchmark path: outputs/training_table_benchmark_1shard')
print('Part count:', len(parts))
print('First 5 parts:')
for p in parts[:5]:
    print(' -', p)

first = pd.read_parquet(parts[0])
print('\nColumns ({}):'.format(len(first.columns)))
print(first.columns.tolist())

print('\nHead (10 rows, selected cols):')
show_cols = [
    'split',
    'session',
    'target',
    'aid',
    'label',
    'candidate_rank',
    'heuristic_score',
    'prefix_len',
    'future_len',
]
print(first[show_cols].head(10).to_string(index=False))

print('\nDistribution (first part):')
print(' split:', first['split'].value_counts().to_dict())
print(' target:', first['target'].value_counts().to_dict())
print(' label mean:', float(first['label'].mean()))

row_count = 0
for p in parts:
    row_count += len(pd.read_parquet(p, columns=['label']))
print('\nTotal rows (all parts):', row_count)
