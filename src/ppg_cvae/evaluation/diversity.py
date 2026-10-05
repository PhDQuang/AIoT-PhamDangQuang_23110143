import numpy as np
from .beats import extract_beats
def aligned_beat_diversity(signals,fs=64,n_points=128,**detector):
    beats=[]
    for s in np.asarray(signals):
        extracted,_=extract_beats(s,fs,**detector)
        for seg in extracted:
            if len(seg)>2:
                u=np.linspace(0,1,n_points); v=np.linspace(0,1,len(seg)); q=np.interp(u,v,seg); span=np.ptp(q)
                if span>1e-8: beats.append((q-q.min())/span)
    if not beats: return {'mean_temporal_std':None,'num_beats':0}
    a=np.asarray(beats); return {'mean_temporal_std':float(a.std(axis=0).mean()),'num_beats':int(len(a))}
