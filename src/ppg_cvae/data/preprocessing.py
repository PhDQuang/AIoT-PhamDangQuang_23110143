"""Leakage-safe continuous filtering, windowing and fold normalization."""
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.signal import butter,sosfiltfilt

@dataclass
class NormalizationStats:
    mean: float; std: float; epsilon: float=1e-8
    def transform(self,x): return (np.asarray(x,dtype=np.float32)-self.mean)/(self.std+self.epsilon)
    def inverse(self,x): return np.asarray(x)*(self.std+self.epsilon)+self.mean

def filter_continuous(ppg,fs=64):
    x=np.asarray(ppg,dtype=float)
    if x.ndim!=1: raise ValueError('PPG segment must be one-dimensional')
    if not np.isfinite(x).all(): raise ValueError('cannot filter segment containing NaN/Inf')
    sos=butter(N=4,Wn=[0.5,8],btype='bandpass',fs=fs,output='sos')
    if len(x)<32: raise ValueError('segment too short for filtfilt')
    return sosfiltfilt(sos,x).astype(np.float32)

def fit_normalization(train_segments:Iterable[np.ndarray],epsilon=1e-8):
    vals=[np.asarray(x,dtype=np.float64).ravel() for x in train_segments]
    if not vals: raise ValueError('no valid train signal points')
    a=np.concatenate(vals); a=a[np.isfinite(a)]
    if a.size==0: raise ValueError('train points are all non-finite')
    std=float(a.std()); return NormalizationStats(float(a.mean()),std if std>epsilon else epsilon,epsilon)

def quality_flags(window,clip_z=1e6):
    x=np.asarray(window)
    reasons=[]
    if x.size==0: return ['empty']
    if not np.isfinite(x).all(): reasons.append('nan_or_inf')
    elif np.ptp(x)<1e-8: reasons.append('nearly_constant')
    if np.isfinite(x).all() and np.sum(np.abs(x)>=clip_z)>0: reasons.append('suspicious_clipping')
    if x.size > 1 and np.isfinite(x).all():
        flat = np.abs(np.diff(x)) <= 1e-8
        edges = np.diff(np.r_[False, flat, False].astype(int))
        if flat.any() and np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) >= 32:
            reasons.append('suspicious_flat_region')
    return reasons

def window_segment(filtered,hr_values=None,start_offset=0,window_size=512,stride=128,subject_id='',fold_id=None,split='',hr_alignment='window',fs=64):
    rows=[]; x=np.asarray(filtered)
    for start in range(0,max(0,len(x)-window_size+1),stride):
        w=x[start:start+window_size]; reasons=quality_flags(w); hr=np.nan
        if hr_values is not None:
            labels=np.asarray(hr_values).reshape(-1)
            expected=len(range(0,max(0,len(x)-window_size+1),stride)) if hr_alignment=='window' else len(x)
            if hr_alignment not in {'window','sample'} or len(labels)!=expected:
                raise ValueError('HR alignment must be explicit and label count must match windows or signal samples')
            hr=float(labels[start//stride] if hr_alignment=='window' else labels[start+window_size//2])
        if not np.isfinite(hr) or hr <= 0: reasons.append('invalid_hr_reference')
        rows.append({"subject_id":subject_id,"fold_id":fold_id,"split":split,"start_sample":start_offset+start,"end_sample":start_offset+start+window_size,"start_time":(start_offset+start)/fs,"hr_reference_bpm":hr,"valid":not reasons,"exclusion_reason":";".join(reasons)})
    return rows


def save_normalization_stats(stats, path):
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'mean': stats.mean, 'std': stats.std, 'epsilon': stats.epsilon}, indent=2), encoding='utf-8')

def load_normalization_stats(path):
    import json
    from pathlib import Path
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    return NormalizationStats(float(d['mean']), float(d['std']), float(d.get('epsilon',1e-8)))
