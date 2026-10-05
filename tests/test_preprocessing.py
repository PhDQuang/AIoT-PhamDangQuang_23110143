import numpy as np
from ppg_cvae.data.preprocessing import fit_normalization

def test_train_stats_only():
 s=fit_normalization([np.array([1.,2.,3.])]); assert abs(s.mean-2)<1e-6; assert abs(s.std-np.std([1.,2.,3.]))<1e-6
