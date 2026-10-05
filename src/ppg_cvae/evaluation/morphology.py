import numpy as np
from .beats import extract_beats
def pulse_features(signal,fs=64,**detector):
    beats, candidates=extract_beats(signal,fs,**detector); vals=[]
    for seg in beats:
        foot=0; footv=float(seg[foot]); peak_index=int(np.argmax(seg)); peak=float(seg[peak_index]); amp=peak-footv
        half=footv+0.5*amp; above=np.flatnonzero(seg>=half)
        if len(above):
            # Use the connected half-height region containing the systolic peak.
            left=peak_index; right=peak_index
            while left>0 and seg[left-1]>=half: left-=1
            while right<len(seg)-1 and seg[right+1]>=half: right+=1
            rise=peak_index/fs; width=(right-left)/fs; cycle=(len(seg)-1)/fs
            vals.append((rise,width,rise/cycle if cycle else np.nan))
    if not vals: return {'rise_time_s':None,'width50_s':None,'rise_cycle_ratio':None,'beat_extraction_success':0.0}
    a=np.asarray(vals); return {'rise_time_s':float(np.median(a[:,0])),'width50_s':float(np.median(a[:,1])),'rise_cycle_ratio':float(np.median(a[:,2])),'beat_extraction_success':float(len(vals)/max(1,candidates))}
