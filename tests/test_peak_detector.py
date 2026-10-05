import numpy as np
from ppg_cvae.evaluation.heart_rate import estimate_hr
def test_flat_invalid():
 assert not estimate_hr(np.zeros(512)).measurable; assert not estimate_hr(np.full(512,np.nan)).measurable
def test_sine():
 t=np.arange(512)/64; x=np.sin(2*np.pi*t); assert estimate_hr(x).measurable
