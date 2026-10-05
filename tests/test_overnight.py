"""Exercise cloud orchestration and its validation-before-test boundary."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ppg_cvae import workflow
from ppg_cvae.data.prepare import prepare_fold
from ppg_cvae.evaluation.calibration import calibrate_detector
from ppg_cvae.training.train_hr_proxy import train_proxy


def test_overnight_reduced_pipeline_locks_protocol_before_test(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('overnight', Path('scripts/run_overnight.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    raw = tmp_path / 'raw'
    raw.mkdir()
    t = np.arange(2048) / 64
    for i in range(1, 16):
        np.savez(raw / f'S{i}.npz', ppg=np.sin(2 * np.pi * t), hr=np.full(13, 60.))
    root = tmp_path / 'artifacts'
    for fold in (1, 2):
        prepared = root / 'prepared' / f'fold_{fold:02d}'
        prepare_fold(raw, prepared, fold, label_start_sample=0, alignment_source='synthetic verification')
        tx, tc = workflow.load_prepared(prepared, 'train')
        vx, vc = workflow.load_prepared(prepared, 'val')
        proxy = root / 'proxy' / f'fold_{fold:02d}'
        train_proxy(tx, tc, vx, vc, epochs=1, output_dir=proxy)
        calibrate_detector(vx, vc, proxy)
    original_train = workflow.train_cvae
    original_tune = workflow.tune_fold
    original_fit = workflow.fit_baseline
    original_evaluate = workflow.evaluate_run
    original_resources = module.measure_resources

    def short_train(*args, **kwargs):
        args[5].max_epochs = 1
        return original_train(*args, **kwargs)

    def short_tune(*args, **kwargs):
        kwargs.update(max_epochs=1, num_val_samples=4)
        return original_tune(*args, **kwargs)

    def short_fit(*args, **kwargs):
        kwargs['max_train_windows'] = 3
        return original_fit(*args, **kwargs)

    def guarded_evaluate(*args, **kwargs):
        locked = json.loads((root / 'overnight/locked_evaluation_protocol.json').read_text())
        assert locked['locked'] and len(locked['folds']) == 2
        assert locked['num_samples'] == 4
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(workflow, 'train_cvae', short_train)
    monkeypatch.setattr(workflow, 'tune_fold', short_tune)
    monkeypatch.setattr(workflow, 'fit_baseline', short_fit)
    monkeypatch.setattr(workflow, 'evaluate_run', guarded_evaluate)
    monkeypatch.setattr(workflow, '_plot_samples', lambda *args: None)
    monkeypatch.setattr(module, 'measure_resources',
                        lambda *args: original_resources(*args, iterations=3, warmup=1))
    module.run_pipeline(root, folds=(1, 2), seeds=(42,), device='cpu', num_samples=4)
    progress = json.loads((root / 'overnight/progress.json').read_text())
    assert progress['status'] == 'complete'
    assert (root / 'figures/cross_fold_generation.png').stat().st_size > 0
    rows = workflow.pd.read_csv(root / 'summary/all_runs.csv')
    assert len(rows) == 24 and rows.groupby('method').fold.nunique().eq(2).all()
    resources = workflow.pd.read_csv(root / 'summary/resource_metrics.csv')
    assert resources.fold.tolist() == [1, 2]
