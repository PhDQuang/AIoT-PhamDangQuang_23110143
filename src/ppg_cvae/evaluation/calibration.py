"""Validation-only detector calibration; persist before examining test."""
import numpy as np
import pandas as pd
from .heart_rate import estimate_hr
from ..utils.io import save_json


def calibrate_detector(val_x, val_c, output_dir, prominences=(None, 0.1, 0.25, 0.5, 1.0)):
    from pathlib import Path
    output_dir = Path(output_dir)
    if not len(val_x) or len(val_x) != len(val_c):
        raise ValueError('Nonempty paired validation signals and labels required')
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for prominence in prominences:
        errors = []
        for x, c in zip(val_x, val_c):
            m = estimate_hr(x, prominence=prominence)
            errors.append(abs(m.hr-float(c)*60) if m.measurable else np.nan)
        errors = np.asarray(errors)
        valid = np.isfinite(errors)
        results.append(dict(prominence=prominence, V=float(valid.mean()),
                            mae_hr=float(errors[valid].mean()) if valid.any() else None,
                            A3=float(np.sum(errors[valid]<=3)/len(errors))))
    if not any(r['mae_hr'] is not None for r in results):
        raise ValueError('Detector cannot measure any real validation window')
    # Fixed rule: prefer V>=95%, then MAE; otherwise maximize all-sample A3.
    eligible = [r for r in results if r['V']>=.95 and r['mae_hr'] is not None]
    best = min(eligible, key=lambda r: r['mae_hr']) if eligible else max(results, key=lambda r: (r['A3'], r['V']))
    detector = dict(min_distance=20, prominence=best['prominence'])
    save_json(detector, output_dir/'detector_config.json')
    save_json({'selection_split': 'val', 'results': results, 'selected': best,
               'rule': 'V>=0.95 then MAE; fallback maximum A3, then V'}, output_dir/'detector_validation_metrics.json')
    return detector
