"""Train-fitted Two-Gaussian waveform synthesis baseline."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares

@dataclass
class GaussianParams:
    a1: float=1.; mu1: float=.18; sigma1: float=.08
    a2: float=.35; mu2: float=.42; sigma2: float=.12; baseline: float=0.


def _pulse(phase, p):
    # Periodic extension avoids a jump when the phase wraps from 1 to 0.
    y = np.full(np.shape(phase), p.baseline, dtype=float)
    for shift in (-2,-1,0,1,2):
        y += p.a1*np.exp(-.5*((phase-p.mu1+shift)/p.sigma1)**2)
        y += p.a2*np.exp(-.5*((phase-p.mu2+shift)/p.sigma2)**2)
    return y


def synthesize_two_gaussian(hr, num_samples=1, fs=64, params=None, seed=None):
    if hr <= 0 or fs <= 0 or num_samples < 1: raise ValueError('Positive HR, fs and sample count required')
    bank = [GaussianParams()] if params is None else ([params] if isinstance(params,GaussianParams) else list(params))
    if not bank: raise ValueError('Empty fitted Gaussian parameter bank')
    rng = np.random.default_rng(seed)
    phase = (np.arange(512)[None,:]/fs/(60./hr)+rng.random(num_samples)[:,None])%1
    selected = rng.integers(len(bank),size=num_samples)
    y = np.stack([_pulse(ph,bank[i]) for ph,i in zip(phase,selected)])
    return y.astype(np.float32)[:,None,:]


def fit_two_gaussian(train_beats):
    """Fit amplitude, locations, widths and baseline jointly for each train beat.

    Returned parameter sets are sampled as whole tuples, preserving relations.
    No defaults are substituted for failed fits or missing real training data.
    """
    fitted = []
    for beat in train_beats:
        y = np.asarray(beat,float).reshape(-1)
        if len(y)<4 or not np.isfinite(y).all() or np.ptp(y)<=1e-8: continue
        phase = np.linspace(0,1,len(y))
        amp, baseline = float(np.ptp(y)),float(y.min())
        peak = float(np.clip(phase[np.argmax(y)],.05,.55))
        initial = [amp,peak,.08,.35*amp,min(.9,peak+.25),.12,baseline]
        def residual(v): return _pulse(phase,GaussianParams(*v))-y
        result = least_squares(residual,initial,
            bounds=([0,.02,.02,0,.25,.02,baseline-amp],
                    [3*amp,.6,.3,3*amp,.98,.3,baseline+amp]),max_nfev=1000,
                    ftol=1e-6,xtol=1e-6,gtol=1e-6)
        if result.success and np.isfinite(result.x).all(): fitted.append(GaussianParams(*result.x))
    if not fitted: raise ValueError('No valid train beats could be fitted')
    return fitted
