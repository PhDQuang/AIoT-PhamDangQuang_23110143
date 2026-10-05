import argparse
from pathlib import Path
from ppg_cvae.evaluation.aggregate import aggregate_runs,summarize_runs
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
rows=aggregate_runs(a.root)
by_fold,across=summarize_runs(rows)
out=Path(a.root)/'summary'; out.mkdir(parents=True,exist_ok=True)
rows.to_csv(out/'all_runs.csv',index=False)
by_fold.to_csv(out/'seed_summary_by_fold.csv',index=False)
across.to_csv(out/'cross_fold_summary.csv',index=False)
print(across.to_string(index=False))
