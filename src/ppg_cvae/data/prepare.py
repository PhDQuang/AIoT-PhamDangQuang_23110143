"""Prepare fold artifacts with explicit label alignment and no gap interpolation."""
from pathlib import Path
import numpy as np
import pandas as pd
from .dalia import discover_subject_files, load_subject, subject_id_from_path
from .manifest import assign_split, make_manifest, save_manifest
from .preprocessing import filter_continuous, quality_flags, fit_normalization, save_normalization_stats
from .splits import get_fold
from ..utils.io import save_json


def prepare_fold(data_root, output_dir, fold_id=1, *, label_start_sample,
                 alignment_source, window_size=512, stride=128, edge_margin=128):
    """label[j] belongs to PPG[s0+j*stride:s0+j*stride+window_size].

    s0 and its provenance must be supplied after inspecting the dataset readme.
    Counts are validated before filtering; exclusions never renumber labels.
    Each finite run is filtered separately. A fixed margin is excluded at both
    ends of every run. Train normalization counts each accepted sample once.
    """
    get_fold(fold_id)
    if label_start_sample < 0 or not alignment_source.strip():
        raise ValueError('Supply a nonnegative label_start_sample and alignment_source')
    if window_size != 512 or stride != 128 or edge_margin < 0:
        raise ValueError('Expected 512-sample windows, stride 128 and nonnegative edge margin')
    files = discover_subject_files(data_root)
    expected = {f'S{i}' for i in range(1, 16)}
    found = {subject_id_from_path(p) for p in files}
    if found != expected:
        raise ValueError(f'Full five-fold study requires S1..S15; missing {sorted(expected-found)}')
    output_dir = Path(output_dir)
    rows, train_points = [], []
    split_x = {s: [] for s in ('train', 'val', 'test')}
    split_c = {s: [] for s in split_x}
    for path in files:
        rec = load_subject(path)
        raw = rec.ppg
        n_windows = max(0, (len(raw)-label_start_sample-window_size)//stride+1)
        if rec.hr is None or len(rec.hr) != n_windows:
            raise ValueError(f'{rec.subject_id}: {n_windows} windows but '
                             f'{None if rec.hr is None else len(rec.hr)} labels; '
                             'verify schema, s0 and stride rather than truncating labels')
        filtered = np.full(raw.shape, np.nan, dtype=np.float32)
        interior = np.zeros(len(raw), dtype=bool)
        edges = np.diff(np.r_[False, np.isfinite(raw), False].astype(int))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b-a >= 32:
                filtered[a:b] = filter_continuous(raw[a:b], rec.ppg_fs)
                interior[a+edge_margin:max(a+edge_margin, b-edge_margin)] = True
        split = assign_split(rec.subject_id, fold_id)
        used = np.zeros(len(raw), dtype=bool)
        for j, hr in enumerate(rec.hr):
            start = label_start_sample+j*stride
            end = start+window_size
            reasons = quality_flags(raw[start:end])
            if not np.isfinite(filtered[start:end]).all(): reasons.append('missing_or_short_segment')
            if not interior[start:end].all(): reasons.append('filter_boundary')
            if not np.isfinite(hr) or hr <= 0: reasons.append('invalid_hr_reference')
            reasons = sorted(set(reasons))
            rows.append(dict(subject_id=rec.subject_id, start_sample=start, end_sample=end,
                             start_time=start/rec.ppg_fs, hr_reference_bpm=float(hr),
                             valid=not reasons, exclusion_reason=';'.join(reasons)))
            if not reasons:
                split_x[split].append(filtered[start:end].copy())
                split_c[split].append(float(hr)/60.)
                if split == 'train': used[start:end] = True
        if split == 'train': train_points.append(filtered[used])
    for split, windows in split_x.items():
        if not windows: raise ValueError(f'No valid {split} windows in fold {fold_id}')
    stats = fit_normalization(train_points)
    manifest = make_manifest(rows, fold_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, output_dir/'manifest.csv')
    save_normalization_stats(stats, output_dir/'normalization_stats.json')
    for split in split_x:
        np.savez_compressed(output_dir/f'{split}.npz',
                            x=stats.transform(np.stack(split_x[split]))[:, None, :],
                            c=np.asarray(split_c[split], dtype=np.float32))
    save_json(dict(fold=fold_id, label_start_sample=label_start_sample,
                   alignment_source=alignment_source, window_size=window_size, stride=stride,
                   edge_margin=edge_margin, sampling_rate=64, normalization='unique valid train points',
                   source_files=[str(p) for p in files]), output_dir/'preprocessing_config.json')
    summary = manifest.groupby(['split', 'subject_id']).agg(
        total_windows=('valid', 'size'), valid_windows=('valid', 'sum'),
        hr_min=('hr_reference_bpm', 'min'), hr_max=('hr_reference_bpm', 'max'))
    summary.to_csv(output_dir/'subject_summary.csv')
    coverage = []
    for split in split_x:
        valid = manifest[(manifest.split == split) & manifest.valid]
        for hr in (60, 80, 120):
            nearby = valid[valid.hr_reference_bpm.between(hr-10, hr+10, inclusive='left')]
            coverage.append(dict(split=split, target_hr=hr, windows=len(nearby), subjects=nearby.subject_id.nunique()))
    pd.DataFrame(coverage).to_csv(output_dir/'hr_coverage.csv', index=False)
    return manifest


def load_prepared(output_dir, split):
    with np.load(Path(output_dir)/f'{split}.npz', allow_pickle=False) as data:
        return data['x'], data['c']
