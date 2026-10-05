from pathlib import Path
import json,pandas as pd
def aggregate_runs(root):
    rows=[]
    for p in Path(root).rglob('generation_summary.csv'): rows.extend(pd.read_csv(p).to_dict('records'))
    return pd.DataFrame(rows)

def summarize_runs(rows):
    """First average seeds within fold, then mean/std across folds."""
    if rows.empty: return pd.DataFrame(),pd.DataFrame()
    keys=['method','fold','target_hr']
    required=set(keys+['seed'])
    if not required.issubset(rows.columns): raise ValueError(f'Missing run identity columns: {required-set(rows.columns)}')
    if rows.duplicated(keys+['seed']).any(): raise ValueError('Duplicate method/fold/seed/HR results')
    metrics=[c for c in ('mae_hr','median_abs_error','V','A3','mean_temporal_std') if c in rows]
    by_fold=rows.groupby(keys,dropna=False)[metrics].agg(['mean','std','count'])
    by_fold.columns=['_'.join(c) for c in by_fold.columns]
    by_fold=by_fold.reset_index()
    means=[f'{c}_mean' for c in metrics]
    across=by_fold.groupby(['method','target_hr'])[means].agg(['mean','std','count'])
    across.columns=['_'.join(c) for c in across.columns]
    return by_fold,across.reset_index()
