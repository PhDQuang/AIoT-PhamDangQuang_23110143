"""Descriptive ACC audit only; never changes prepared data or model selection."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from ppg_cvae.data.dalia import discover_subject_files, load_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/kaggle/input')
    parser.add_argument('--output', default='artifacts/completion/motion')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in discover_subject_files(args.data_root):
        rec = load_subject(path)
        if rec.acc is None or rec.acc.ndim != 2 or rec.acc.shape[1] != 3:
            raise ValueError(f'{rec.subject_id}: synchronized wrist ACC Nx3 required')
        # Official synchronized data: BVP 64 Hz, wrist ACC 32 Hz, t=0 common.
        if abs(len(rec.acc) / 32 - len(rec.ppg) / 64) > 1:
            raise ValueError(f'{rec.subject_id}: ACC/PPG duration mismatch')
        n = (len(rec.ppg) - 512) // 128 + 1
        if rec.hr is None or len(rec.hr) != n:
            raise ValueError(f'{rec.subject_id}: label count disagrees with s0=0')
        for j in range(n):
            acc = rec.acc[j*64:j*64+256].astype(np.float64)
            finite = len(acc) == 256 and np.isfinite(acc).all()
            rms = float(np.sqrt(np.mean(np.sum((acc-acc.mean(0))**2, axis=1)))) if finite else np.nan
            rows.append(dict(subject_id=rec.subject_id, start_sample=j*128,
                             end_sample=j*128+512, motion_rms_g=rms,
                             acc_finite=finite))
        print(rec.subject_id, n, flush=True)
    pd.DataFrame(rows).to_csv(out/'motion_windows.csv', index=False)
    (out/'motion_definition.json').write_text(json.dumps({
        'feature': 'sqrt(mean(sum((ACC-axis_mean)^2))) over 256 wrist ACC samples',
        'unit': 'g as stored in the synchronized PPG-DaLiA wrist ACC',
        'acc_fs': 32, 'ppg_fs': 64, 'label_start_sample': 0,
        'alignment': 'common t=0 in synchronized SX.pkl; exact start/end converted by fs ratio',
        'role': 'post-hoc descriptive audit; no filtering or model-selection changes',
        'warning': 'motion level is not a clean-PPG quality label',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
