"""Independent peak detector and generation metrics."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.signal import find_peaks
@dataclass
class HRMeasurement:
    hr:float|None; measurable:bool; peaks:int; reason:str

def estimate_hr(signal,fs=64,min_distance=20,prominence=None):
    x=np.asarray(signal,dtype=float).squeeze()
    if x.ndim!=1 or not np.isfinite(x).all(): return HRMeasurement(None,False,0,'invalid_signal')
    if x.size == 0 or np.ptp(x)<1e-8: return HRMeasurement(None,False,0,'flat_signal')
    kwargs={'distance':min_distance}
    if prominence is not None: kwargs['prominence']=prominence
    peaks,_=find_peaks(x,**kwargs)
    if len(peaks)<3: return HRMeasurement(None,False,len(peaks),'fewer_than_3_peaks')
    ibi=np.diff(peaks)/fs; hr=float(60.0/np.median(ibi)) if np.all(ibi>0) else np.nan
    if not np.isfinite(hr) or not 40<=hr<=180: return HRMeasurement(None,False,len(peaks),'hr_out_of_range')
    return HRMeasurement(hr,True,len(peaks),'ok')

def generation_metrics(signals,target_hr,fs=64,**kwargs):
    rows=[]
    for i,s in enumerate(np.asarray(signals)):
        m=estimate_hr(s,fs,**kwargs); err=abs(m.hr-target_hr) if m.measurable else np.nan
        rows.append({'index':i,'target_hr':target_hr,'measured_hr':m.hr,'measurable':m.measurable,'abs_error':err,'failure_reason':m.reason})
    errs=np.array([r['abs_error'] for r in rows],float); ok=np.isfinite(errs)
    return rows,{'target_hr':target_hr,'mae_hr':float(np.mean(errs[ok])) if ok.any() else None,'median_abs_error':float(np.median(errs[ok])) if ok.any() else None,'V':float(ok.mean()) if len(ok) else 0.0,'A3':float(np.sum(errs[ok]<=3)/len(rows)) if rows else 0.0,'failed_windows':int((~ok).sum()),'num_samples':len(rows)}
