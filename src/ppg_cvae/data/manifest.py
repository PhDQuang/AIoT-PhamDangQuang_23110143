from pathlib import Path
import pandas as pd
from .splits import get_fold

def assign_split(subject,fold_id):
    f=get_fold(fold_id)
    return 'train' if subject in f.train_subjects else 'val' if subject in f.val_subjects else 'test' if subject in f.test_subjects else 'unknown'
def make_manifest(rows,fold_id):
    df=pd.DataFrame(rows)
    if not df.empty: df['split']=df['subject_id'].map(lambda s:assign_split(s,fold_id)); df['fold_id']=fold_id
    required=['subject_id','fold_id','split','start_sample','end_sample','start_time','hr_reference_bpm','valid','exclusion_reason']
    for c in required:
        if c not in df: df[c]=None
    return df[required]
def save_manifest(df,path): Path(path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
