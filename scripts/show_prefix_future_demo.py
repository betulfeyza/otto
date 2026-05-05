import glob
import pandas as pd

smoke_part = sorted(glob.glob('outputs/training_table_smoke/part-*.parquet'))[0]
cutoff_ts = int(pd.read_parquet(smoke_part, columns=['cutoff_ts'])['cutoff_ts'].iloc[0])

train_shard = sorted(glob.glob('archive/train_parquet/*.parquet'))[0]
df = pd.read_parquet(train_shard, columns=['session', 'aid', 'ts', 'type']).sort_values(['session', 'ts'])

agg = df.groupby('session')['ts'].agg(['min', 'max']).reset_index()
sel = agg[(agg['min'] <= cutoff_ts) & (agg['max'] > cutoff_ts)].head(1)
if sel.empty:
    print('No crossing session found in this shard')
    raise SystemExit(0)

session_id = int(sel.iloc[0]['session'])
sess = df[df['session'] == session_id].sort_values('ts').copy()
prefix_df = sess[sess['ts'] <= cutoff_ts]
future_df = sess[sess['ts'] > cutoff_ts]

print('cutoff_ts =', cutoff_ts)
print('session   =', session_id)
print('total events  =', len(sess))
print('prefix events =', len(prefix_df), '(ts <= cutoff)')
print('future events =', len(future_df), '(ts > cutoff)')

print('\n--- FULL SESSION (with split tag) ---')
show = sess[['session', 'ts', 'type', 'aid']].copy()
show['split_side'] = ['PREFIX' if int(x) <= cutoff_ts else 'FUTURE' for x in show['ts']]
print(show.to_string(index=False))

print('\n--- PREFIX ONLY ---')
print(prefix_df[['session', 'ts', 'type', 'aid']].to_string(index=False))

print('\n--- FUTURE ONLY ---')
print(future_df[['session', 'ts', 'type', 'aid']].to_string(index=False))
