"""Run the remaining fixed experiment protocol entirely inside one Kaggle job."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
import torch
from ppg_cvae import workflow
from ppg_cvae.evaluation.aggregate import aggregate_runs, summarize_runs
from ppg_cvae.evaluation.resources import measure_resources
from ppg_cvae.models.hr_proxy import HRProxy
from ppg_cvae.utils.io import save_json

SEEDS = (42, 43, 44)
DATASETS = ('ppg-cvae-prepared-fold01', 'ppg-cvae-prepared-folds02-05')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_bundle(bundle, root):
    manifest = json.loads((bundle / 'bundle_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files_sha256'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
            raise ValueError(f'Invalid bundle checksum/path: {relative}')
    for name in ('proxy', 'tuning', 'cvae', 'gaussian', 'ablations'):
        if (bundle / name).is_dir():
            shutil.copytree(bundle / name, root / name, dirs_exist_ok=True)
    save_json(manifest, root / 'overnight' / 'input_manifest.json')


def import_prepared(root, folds, prepared_root=None):
    mounts = [Path(prepared_root)] if prepared_root else [
        p for slug in DATASETS for p in Path('/kaggle/input').rglob(slug) if p.is_dir()
    ]
    if not mounts:
        raise FileNotFoundError('Attach both prepared Kaggle datasets to the overnight job.')
    required = ('train.npz', 'val.npz', 'test.npz', 'manifest.csv',
                'normalization_stats.json', 'preprocessing_config.json', 'hr_coverage.csv')
    for fold in folds:
        name = f'fold_{fold:02d}'
        destination = root / 'prepared' / name
        sources = [p.parent for mount in mounts for p in mount.rglob('train.npz')
                   if p.parent.name == name and all((p.parent / f).is_file() for f in required)]
        if len(sources) > 1:
            raise ValueError(f'Ambiguous prepared sources for {name}: {sources}')
        if sources:
            if sources[0].resolve() != destination.resolve():
                shutil.copytree(sources[0], destination, dirs_exist_ok=True)
        else:
            prefix = f'prepared/{name}/'
            matches = []
            for mount in mounts:
                for archive in mount.rglob('*.zip'):
                    with ZipFile(archive) as z:
                        if all(prefix + f in z.namelist() for f in required):
                            matches.append(archive)
            if len(matches) != 1:
                raise FileNotFoundError(f'Expected exactly one prepared directory/ZIP for {name}')
            destination.mkdir(parents=True, exist_ok=True)
            with ZipFile(matches[0]) as z:
                for filename in (*required, 'subject_summary.csv'):
                    if prefix + filename in z.namelist():
                        with z.open(prefix + filename) as source, (destination / filename).open('wb') as target:
                            shutil.copyfileobj(source, target)
        if any((destination / filename).stat().st_size == 0 for filename in required):
            raise ValueError(f'Empty prepared input in {name}')


def lock_protocol(root, folds, seeds, num_samples):
    """Snapshot all validation decisions before any model evaluation reads test."""
    records = []
    for fold in folds:
        selected_path = root / 'tuning' / f'fold_{fold:02d}' / 'selected_config.json'
        selected = json.loads(selected_path.read_text(encoding='utf-8'))
        if selected['fold'] != fold or (selected['beta'], selected['lambda_hr']) not in workflow.CANDIDATES:
            raise ValueError(f'Invalid selected candidate for fold {fold}')
        detector_path = root / 'proxy' / f'fold_{fold:02d}' / 'detector_config.json'
        required = []
        for seed in seeds:
            required.extend([root / 'cvae' / f'fold_{fold:02d}' / f'seed_{seed}' / 'best_model.pt',
                             root / 'gaussian' / f'fold_{fold:02d}' / f'seed_{seed}' / 'gaussian_fit.json'])
        for name in ('no_hr_loss', 'no_condition'):
            required.append(root / 'ablations' / name / f'fold_{fold:02d}' / 'seed_42' / 'best_model.pt')
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        records.append(dict(fold=fold, selected=selected,
                            detector=json.loads(detector_path.read_text(encoding='utf-8')),
                            detector_sha256=sha256(detector_path),
                            normalization_sha256=sha256(root / 'prepared' / f'fold_{fold:02d}' / 'normalization_stats.json'),
                            checkpoints_sha256={str(p.relative_to(root)): sha256(p) for p in required}))
    save_json(dict(locked=True, locked_at_unix=time.time(), targets=list(workflow.TARGETS),
                   seeds=list(seeds), num_samples=num_samples, validation_latent_seed=1042,
                   candidates=list(workflow.CANDIDATES), folds=records,
                   proxy_policy='frozen train-fitted proxy selected on real validation',
                   interpretation='HR detector/proxy limitations retained; no tuning after test'),
              root / 'overnight' / 'locked_evaluation_protocol.json')


def plot_summary(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, label in zip(axes, ('mae_hr', 'V', 'A3'), ('HR MAE (bpm)', 'Measurable fraction', 'A3')):
        for method, group in summary.groupby('method'):
            group = group.sort_values('target_hr')
            ax.errorbar(group.target_hr, group[f'{metric}_mean_mean'],
                        yerr=group[f'{metric}_mean_std'].fillna(0), marker='o', capsize=3, label=method)
        ax.set(xlabel='Target HR (bpm)', ylabel=label)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'cross_fold_generation.png', dpi=200)
    plt.close(fig)


def run_pipeline(root, folds=tuple(range(1, 6)), seeds=SEEDS, device='cuda', num_samples=1000):
    root = Path(root)
    progress_path = root / 'overnight' / 'progress.json'
    progress = dict(status='running', started_at_unix=time.time(), completed_stages=[])

    def stage(name, action):
        progress.update(current_stage=name, updated_at_unix=time.time())
        save_json(progress, progress_path)
        print(f'=== START {name} ===', flush=True)
        result = action()
        progress['completed_stages'].append(name)
        save_json(progress, progress_path)
        print(f'=== COMPLETE {name} ===', flush=True)
        if device == 'cuda':
            torch.cuda.empty_cache()
        return result

    try:
        # Every tuning decision is made using train/validation before test access.
        for fold in folds:
            selected = root / 'tuning' / f'fold_{fold:02d}' / 'selected_config.json'
            if not selected.is_file():
                quality = root / 'proxy' / f'fold_{fold:02d}'
                save_json(dict(proxy=json.loads((quality / 'hr_proxy_validation_metrics.json').read_text()),
                               detector=json.loads((quality / 'detector_validation_metrics.json').read_text()),
                               interpretation='exploratory; real validation measurement error retained'),
                          selected.parent / 'measurement_quality.json')
                stage(f'tune_fold_{fold}', lambda f=fold: workflow.tune_fold(root, f, device=device, include_no_hr_control=True))
        for fold in folds:
            for seed in seeds:
                run = root / 'cvae' / f'fold_{fold:02d}' / f'seed_{seed}'
                if not (run / 'best_model.pt').is_file():
                    stage(f'train_fold_{fold}_seed_{seed}', lambda f=fold, s=seed: workflow.train_selected(root, f, s, device=device))
            baseline = root / 'gaussian' / f'fold_{fold:02d}' / 'seed_42'
            if not (baseline / 'gaussian_fit.json').is_file():
                stage(f'gaussian_fold_{fold}', lambda f=fold: workflow.fit_baseline(root, f))
            for seed in seeds:
                target = root / 'gaussian' / f'fold_{fold:02d}' / f'seed_{seed}'
                target.mkdir(parents=True, exist_ok=True)
                if not (target / 'gaussian_fit.json').is_file():
                    shutil.copy2(baseline / 'gaussian_fit.json', target / 'gaussian_fit.json')
            for name in ('no_hr_loss', 'no_condition'):
                run = root / 'ablations' / name / f'fold_{fold:02d}' / 'seed_42'
                if not (run / 'best_model.pt').is_file():
                    stage(f'{name}_fold_{fold}', lambda f=fold, a=name: workflow.train_selected(root, f, 42, device=device, ablation=a))
        stage('lock_evaluation_protocol', lambda: lock_protocol(root, folds, seeds, num_samples))
        for fold in folds:
            for method in ('cvae', 'gaussian', 'no_hr_loss', 'no_condition'):
                for seed in ((42,) if method in ('no_hr_loss', 'no_condition') else seeds):
                    stage(f'evaluate_{method}_fold_{fold}_seed_{seed}',
                          lambda f=fold, s=seed, m=method: workflow.evaluate_run(root, f, s, m, device=device, num_samples=num_samples))
        rows = aggregate_runs(root)
        by_fold, summary = summarize_runs(rows)
        expected_rows = len(folds) * (2 * len(seeds) + 2) * len(workflow.TARGETS)
        if len(rows) != expected_rows:
            raise ValueError(f'Incomplete evaluation: expected {expected_rows} rows, found {len(rows)}')
        directory = root / 'summary'
        directory.mkdir(parents=True, exist_ok=True)
        rows.to_csv(directory / 'all_runs.csv', index=False)
        by_fold.to_csv(directory / 'seed_summary_by_fold.csv', index=False)
        summary.to_csv(directory / 'cross_fold_summary.csv', index=False)
        stage('summary_figures', lambda: plot_summary(summary, root / 'figures'))
        resource_rows = []
        for fold in folds:
            proxy = HRProxy()
            proxy.load_state_dict(torch.load(root / 'proxy' / f'fold_{fold:02d}' / 'hr_proxy.pt', map_location='cpu', weights_only=True))
            for seed in seeds:
                run = root / 'cvae' / f'fold_{fold:02d}' / f'seed_{seed}'
                model, _ = workflow.load_run(run, device='cpu')
                decoder = run / 'decoder_state_dict.pt'
                torch.save(model.decoder.state_dict(), decoder)
                metrics = stage(f'benchmark_fold_{fold}_seed_{seed}', lambda: measure_resources(model, proxy, decoder))
                metrics.update(fold=fold, seed=seed, checkpoint=str(run / 'best_model.pt'),
                               scope='Kaggle host CPU FP32 single-thread; not laptop or microcontroller latency')
                save_json(metrics, run / 'resource_metrics.json')
                resource_rows.append(metrics)
        pd.DataFrame(resource_rows).to_csv(directory / 'resource_metrics.csv', index=False)
        progress.update(status='complete', current_stage=None, finished_at_unix=time.time(),
                        elapsed_minutes=(time.time() - progress['started_at_unix']) / 60)
        save_json(progress, progress_path)
        print('OVERNIGHT PIPELINE COMPLETE: training, test evaluation, figures and CPU benchmark saved.', flush=True)
    except Exception:
        progress.update(status='error', traceback=traceback.format_exc(), updated_at_unix=time.time())
        save_json(progress, progress_path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts')
    parser.add_argument('--bundle', default='inputs/overnight')
    parser.add_argument('--prepared-root')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Submit this overnight job with --gpu T4.')
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    import_bundle(Path(args.bundle), root)
    import_prepared(root, range(1, 6), args.prepared_root)
    torch.set_num_threads(2)
    run_pipeline(root)


if __name__ == '__main__':
    main()
