"""Best-effort PPG-DaLiA discovery with explicit schema validation."""
from dataclasses import dataclass
from pathlib import Path
import pickle, re, numpy as np

@dataclass
class SubjectRecord:
    subject_id:str; ppg:np.ndarray; hr:np.ndarray|None=None; ecg:np.ndarray|None=None; acc:np.ndarray|None=None; ppg_fs:float=64.0

def _find_value(obj,keys):
    if not isinstance(obj,dict): return None
    for k in keys:
        if k in obj: return obj[k]
    return None

def subject_id_from_path(path):
    """Match a complete subject token; S10 must never match S1."""
    path = Path(path)
    for token in (path.stem, path.parent.name):
        match = re.search(r'(?<![A-Z0-9])S(1[0-5]|[1-9])(?![A-Z0-9])', token.upper())
        if match:
            return f'S{int(match.group(1))}'
    raise ValueError(f'Cannot identify S1..S15 from {path}')

def discover_subject_files(root):
    root=Path(root)
    if not root.exists(): raise FileNotFoundError(f'DATA_ROOT does not exist: {root}')
    files=[]
    for p in root.rglob('*'):
        if p.suffix.lower() in {'.pkl','.pickle','.mat','.npz'}:
            try: subject_id_from_path(p)
            except ValueError: continue
            files.append(p)
    if not files: raise FileNotFoundError(f'No supported PPG-DaLiA subject files found below {root}. Expected subject files containing S1..S15.')
    by_subject = {}
    for p in files:
        sid = subject_id_from_path(p)
        if sid in by_subject:
            raise ValueError(f'Duplicate files for {sid}: {by_subject[sid]} and {p}; select one dataset root')
        by_subject[sid] = p
    return sorted(files, key=lambda p: int(subject_id_from_path(p)[1:]))

def load_subject(path,subject_id=None):
    path=Path(path); sid=subject_id or subject_id_from_path(path)
    if path.suffix.lower() in {'.pkl','.pickle'}:
        try:
            with path.open('rb') as f: obj=pickle.load(f)
        except (UnicodeDecodeError, TypeError):
            with path.open('rb') as f: obj=pickle.load(f, encoding='latin1')
    elif path.suffix.lower()=='.npz': obj=dict(np.load(path,allow_pickle=True))
    elif path.suffix.lower()=='.mat':
        from scipy.io import loadmat
        obj=loadmat(path, simplify_cells=True)
    else: raise ValueError(f'unsupported subject file: {path}')
    signal=obj.get('signal') if isinstance(obj,dict) else None
    wrist=signal.get('wrist') if isinstance(signal,dict) else None
    chest=signal.get('chest') if isinstance(signal,dict) else None
    ppg=None
    if isinstance(wrist,dict): ppg=_find_value(wrist,['BVP','bvp','ppg','PPG','wrist_ppg','wrist_bvp'])
    if ppg is None: ppg=_find_value(obj,['ppg','bvp','BVP','wrist_ppg','wrist_bvp'])
    if ppg is None and wrist is not None and not isinstance(wrist,dict):
        w=np.asarray(wrist); ppg=w[:,0] if w.ndim==2 else w
    if ppg is None: raise ValueError(f'{path} recognized but has no PPG/BVP key; inspect available keys: {list(obj.keys()) if isinstance(obj,dict) else type(obj)}')
    hr=_find_value(obj,['label','hr','HR','heart_rate','label_hr','reference_hr'])
    if hr is None and isinstance(signal,dict): hr=_find_value(signal,['label','hr','HR'])
    ecg=_find_value(chest,['ECG','ecg']) if isinstance(chest,dict) else _find_value(obj,['ecg','ECG'])
    acc=_find_value(wrist,['ACC','acc','wrist_acc']) if isinstance(wrist,dict) else _find_value(obj,['acc','ACC','wrist_acc'])
    if acc is None and wrist is not None and not isinstance(wrist,dict):
        w=np.asarray(wrist); acc=w[:,1:4] if w.ndim==2 and w.shape[1]>=4 else None
    ppg = np.asarray(ppg,dtype=np.float32).squeeze()
    hr = None if hr is None else np.asarray(hr,dtype=np.float32).reshape(-1)
    if ppg.ndim != 1 or not ppg.size:
        raise ValueError(f'{path}: expected nonempty one-channel PPG, got {ppg.shape}')
    return SubjectRecord(sid,ppg,hr,None if ecg is None else np.asarray(ecg,dtype=np.float32).squeeze(),None if acc is None else np.asarray(acc,dtype=np.float32),64.0)

def load_all(root): return [load_subject(p) for p in discover_subject_files(root)]
