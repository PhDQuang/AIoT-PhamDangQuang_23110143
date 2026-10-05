"""Complete descriptive tables and local CPU evidence from locked artifacts."""
from pathlib import Path
import argparse
import ctypes
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ppg_cvae.models.cvae import ConditionalVAE, Decoder
from ppg_cvae.evaluation.heart_rate import estimate_hr

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'
OUT = ROOT/'artifacts/completion'
FIG = ROOT/'reports/work/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def save(name):
    plt.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close()


def process_memory():
    if sys.platform != 'win32':
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (k, ctypes.c_size_t) for k in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes': counters.WorkingSetSize,
            'process_peak_working_set_bytes': counters.PeakWorkingSetSize}


def tables(motion):
    overview, bins, support, reasons, motions = [], [], [], [], []
    for f in range(1, 6):
        m = pd.read_csv(ART/f'prepared/fold_{f:02}/manifest.csv')
        train = m[(m.split == 'train') & m.valid]
        if motion:
            acc = pd.read_csv(motion)
            m = m.merge(acc, on=['subject_id', 'start_sample', 'end_sample'], how='left',
                        validate='one_to_one', indicator=True)
            if not m._merge.eq('both').all():
                raise ValueError('ACC audit failed to cover every manifest row')
            thresholds = np.quantile(m.loc[m.split.eq('train'), 'motion_rms_g'].dropna(), [1/3, 2/3])
            m['motion_level'] = pd.cut(m.motion_rms_g, [-np.inf, *thresholds, np.inf],
                                      labels=['low', 'medium', 'high']).astype('object').fillna('unknown')
            (OUT/f'motion_thresholds_fold_{f:02}.json').write_text(json.dumps({
                'thresholds_g': thresholds.tolist(), 'fit_split': 'train',
                'definition': 'tertiles of candidate train-window demeaned ACC RMS',
                'role': 'descriptive post-hoc, not quality filtering'}), encoding='utf-8')
        for split, g in m.groupby('split'):
            valid = g[g.valid]
            overview.append(dict(fold=f, split=split, subjects_before=g.subject_id.nunique(),
                subjects_after=valid.subject_id.nunique(), windows_before=len(g),
                windows_after=len(valid), rejected=(~g.valid).sum(), hr_min=valid.hr_reference_bpm.min(),
                hr_max=valid.hr_reference_bpm.max()))
            for lo in range(0, 200, 20):
                inbin = valid[valid.hr_reference_bpm.between(lo, lo+20, inclusive='left')]
                bins.append(dict(fold=f, split=split, hr_lower=lo, hr_upper=lo+20,
                                 windows=len(inbin), subjects=inbin.subject_id.nunique()))
            for h in [60, 80, 120]:
                near = valid[valid.hr_reference_bpm.between(h-10, h+10, inclusive='left')]
                support.append(dict(fold=f, split=split, target_hr=h, windows=len(near),
                    subjects=near.subject_id.nunique(), outside_train=not train.hr_reference_bpm.min() <= h <= train.hr_reference_bpm.max(),
                    coverage_note='report counts; no pre-registered numerical thinness cutoff'))
            for reason in sorted(set(';'.join(g.exclusion_reason.dropna()).split(';')) - {''}):
                hit = g.exclusion_reason.fillna('').str.split(';').map(lambda rs: reason in rs)
                reasons.append(dict(fold=f, split=split, reason=reason, rejected_windows=hit.sum(),
                                    candidate_windows=len(g), rate=hit.mean()))
            if motion:
                for level, mg in g.groupby('motion_level'):
                    motions.append(dict(fold=f, split=split, motion_level=level,
                        windows_before=len(mg), windows_after=mg.valid.sum(),
                        rejection_rate=(~mg.valid).mean(), motion_rms_median_g=mg.motion_rms_g.median()))
                    for reason in sorted(set(';'.join(mg.exclusion_reason.dropna()).split(';')) - {''}):
                        hit = mg.exclusion_reason.fillna('').str.split(';').map(lambda rs: reason in rs)
                        reasons.append(dict(fold=f, split=split, motion_level=level, reason=reason,
                            rejected_windows=hit.sum(), candidate_windows=len(mg), rate=hit.mean()))
    frames = {'data_by_split': overview, 'hr_bins20': bins, 'target_support': support,
              'exclusions_by_reason': reasons, 'motion_by_split': motions}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    s = pd.DataFrame(support).query("split == 'train'").drop(columns='split').rename(
        columns={'windows': 'train_near_windows', 'subjects': 'train_near_subjects'})
    runs = pd.read_csv(ART/'summary/all_runs.csv').merge(s, on=['fold', 'target_hr'], validate='many_to_one')
    runs.to_csv(OUT/'generation_with_support.csv', index=False)
    summary = runs.groupby(['method', 'target_hr']).agg(
        median_error_mean=('median_abs_error', 'mean'), failed_total=('failed_windows', 'sum'),
        windows_total=('num_samples', 'sum')).reset_index()
    summary.to_csv(OUT/'median_and_failures.csv', index=False)
    persons = pd.concat([pd.read_csv(ART/f'proxy/fold_{f:02}/validation_by_subject_id.csv').assign(fold=f)
                         for f in range(1, 6)], ignore_index=True)
    persons.to_csv(OUT/'real_validation_by_subject.csv', index=False)
    macro = persons.groupby('fold')[['proxy_mae_bpm', 'peak_mae_bpm', 'peak_V']].mean().reset_index()
    macro.to_csv(OUT/'real_validation_subject_macro.csv', index=False)
    return frames


def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dec = Decoder().eval()
    path = ART/'cvae/fold_01/seed_42/decoder_state_dict.pt'
    dec.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    detector = json.loads((ART/'proxy/fold_01/detector_config.json').read_text())
    machine = json.loads((ROOT/'reports/work/local_machine.json').read_text(encoding='utf-8-sig'))
    rows = []
    z, c = torch.randn(1, 16), torch.ones(1)
    with torch.inference_mode():
        for _ in range(100): dec(z, c)
        for h in [60, 80, 120]:
            c.fill_(h/60)
            decoder_times, pipeline_times = [], []
            for _ in range(1000):
                start = time.perf_counter_ns(); dec(z, c)
                decoder_times.append((time.perf_counter_ns()-start)/1e6)
            for _ in range(1000):
                start = time.perf_counter_ns()
                signal = dec(torch.randn(1, 16), c).numpy()[0, 0]
                estimate_hr(signal, **detector)
                pipeline_times.append((time.perf_counter_ns()-start)/1e6)
            rows.append(dict(target_hr=h, decoder_median_ms=np.median(decoder_times),
                decoder_p95_ms=np.percentile(decoder_times, 95),
                pipeline_median_ms=np.median(pipeline_times), pipeline_p95_ms=np.percentile(pipeline_times, 95)))
    pd.DataFrame(rows).to_csv(OUT/'local_cpu_latency.csv', index=False)
    metadata = dict(**machine, python=sys.version, torch=str(torch.__version__),
        runtime_os=platform.platform(), device='cpu', dtype='float32', num_threads=1,
        batch_size=1, warmup=100, iterations_per_hr=1000, checkpoint=str(path.relative_to(ROOT)),
        decoder_bytes=path.stat().st_size, memory=process_memory(),
        memory_scope='whole Python process incl. PyTorch, pandas, matplotlib and analysis; not decoder-only RAM',
        pipeline_scope='sample z + decoder + numpy view + estimate_hr; excludes model loading, IO, plotting',
        benchmark_scope='local Windows host; separate from original Kaggle experiment')
    (OUT/'local_cpu_environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


def figures(frames):
    macro = pd.read_csv(OUT/'real_validation_subject_macro.csv')
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].bar(macro.fold-.18, macro.proxy_mae_bpm, .36, color='.3', label='Proxy')
    axs[0].bar(macro.fold+.18, macro.peak_mae_bpm, .36, color='.8', edgecolor='black', label='Detector')
    axs[0].set(xlabel='Fold', ylabel='MAE macro theo người (bpm)', xticks=range(1, 6)); axs[0].legend()
    axs[1].plot(macro.fold, 100*macro.peak_V, 'o-', color='black')
    axs[1].set(xlabel='Fold', ylabel='Detector V macro theo người (%)', xticks=range(1, 6), ylim=(0, 105))
    fig.tight_layout(); save('proxy')
    # Shared random draws across conditions: generated index 0 has the same z.
    fig, axs = plt.subplots(3, 2, figsize=(9, 5))
    t = np.arange(512)/64
    for i, h in enumerate([60, 80, 120]):
        wave = np.load(ART/f'cvae/fold_01/seed_42/generated_{h}.npz')['ppg'][0, 0]
        axs[i, 0].plot(t, wave, color='black'); axs[i, 0].set_title(f'Cùng z index 0 | HR={h}')
        wave = np.load(ART/'cvae/fold_01/seed_42/generated_80.npz')['ppg'][i, 0]
        axs[i, 1].plot(t, wave, color='black'); axs[i, 1].set_title(f'Cùng HR=80 | z index {i}')
    for ax in axs.flat: ax.set(xlabel='Thời gian (s)', ylabel='Biên độ Z')
    fig.tight_layout(); save('waveforms')
    # Validity and counterexamples use predetermined index 0 / first recorded failure.
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    runs = pd.read_csv(ART/'summary/cross_fold_summary.csv')
    for method, color in [('cvae', 'black'), ('gaussian', '.5')]:
        g = runs[runs.method.eq(method)].sort_values('target_hr')
        axs[0].plot(g.target_hr, 100*g.V_mean_mean, 'o-', color=color, label=method)
    axs[0].set(ylim=(0, 105), xlabel='HR yêu cầu', ylabel='V (%)'); axs[0].legend()
    wave = np.load(ART/'cvae/fold_01/seed_42/generated_120.npz')['ppg'][0, 0]
    detector = json.loads((ART/'proxy/fold_01/detector_config.json').read_text())
    measured = estimate_hr(wave, **detector)
    axs[1].plot(t, wave, color='black'); axs[1].set_title(f'cVAE 120 | đo {measured.hr:.1f}\nindex 0, đo được nhưng sai HR')
    failures = None
    for p in sorted((ART/'gaussian').glob('fold_*/seed_*/generation_samples.csv')):
        g = pd.read_csv(p); failed = g[~g.measurable]
        if len(failed): failures = (p.parent, failed.iloc[0]); break
    if failures is None: raise ValueError('No non-measurable example available')
    directory, row = failures
    wave = np.load(directory/f'generated_{int(row.target_hr)}.npz')['ppg'][int(row['index']), 0]
    axs[2].plot(t, wave, color='black')
    axs[2].set_title(f'Gaussian HR={int(row.target_hr)}\nKhông đo được: {row.failure_reason}')
    for ax in axs[1:]: ax.set(xlabel='Thời gian (s)', ylabel='Biên độ Z')
    fig.tight_layout(); save('validity')
    (OUT/'failure_example.json').write_text(json.dumps({'method': 'gaussian', 'run': str(directory.relative_to(ART)),
        'target_hr': int(row.target_hr), 'index': int(row['index']), 'reason': row.failure_reason}, indent=2), encoding='utf-8')
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    samples = pd.read_csv(ART/'cvae/fold_01/seed_42/generation_samples.csv')
    allruns = pd.read_csv(ART/'summary/all_runs.csv')
    for j, h in enumerate([60, 80, 120]):
        axs[0, j].hist(samples[samples.target_hr.eq(h)].measured_hr, bins=np.arange(40, 185, 5), color='.5')
        axs[0, j].axvline(h, color='black', ls='--'); axs[0, j].set(title=f'Fold 1 seed 42 | h={h}', xlabel='HR đo được', ylabel='Cửa sổ')
        for seed, ls in [(42, 'o-'), (43, 's--'), (44, '^:')]:
            g = allruns.query('method == "cvae" and target_hr == @h and seed == @seed').sort_values('fold')
            axs[1, j].plot(g.fold, g.mae_hr, ls, color={42: 'black', 43: '.4', 44: '.7'}[seed], label=f'Seed {seed}')
        axs[1, j].set(xlabel='Fold', ylabel='MAE (bpm)', xticks=range(1, 6))
    axs[1, 0].legend(fontsize=7); fig.tight_layout(); save('hr_distribution')
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    cf = pd.read_csv(ART/'cvae/fold_01/seed_42/morphology_comparison.csv')
    gf = pd.read_csv(ART/'gaussian/fold_01/seed_42/morphology_comparison.csv')
    for ax, feature, label in zip(axs.flat, ['rise_time_s', 'width50_s', 'rise_cycle_ratio',
        'beat_extraction_success', 'amplitude_std', 'amplitude_range'],
        ['Rise (s)', 'Width50 (s)', 'Rise / cycle', 'Trích nhịp thành công', 'Amplitude SD (Z)', 'Peak-to-peak (Z)']):
        for df, source, color, ls, name in [(cf, 'real_test', 'black', 'o-', 'Thật'),
            (cf, 'generated', '.4', 's--', 'cVAE'), (gf, 'generated', '.7', '^:', 'Gaussian')]:
            g = df[(df.feature == feature) & (df.source == source)].sort_values('target_hr')
            ax.errorbar(g.target_hr, g['median'], yerr=[g['median']-g.q25, g.q75-g['median']],
                        fmt=ls, color=color, capsize=2, label=name)
        ax.set(title=label, xticks=[60, 80, 120], xlabel='HR mục tiêu')
        if feature == 'beat_extraction_success':
            ax.set_ylim(0, 1.05)
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); save('morphology')
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    bins = pd.DataFrame(frames['hr_bins20'])
    for ax, split in zip(axs, ['train', 'val', 'test']):
        for f in range(1, 6):
            g = bins.query('fold == @f and split == @split')
            ax.plot(g.hr_lower+10, g.windows, label=f'F{f}', color=str(.1+(f-1)*.17))
        ax.set(title=f'{split} | bin 20 bpm', xlabel='HR giữa bin', ylabel='Cửa sổ hợp lệ', xlim=(30, 190))
    axs[0].legend(fontsize=7); fig.tight_layout(); save('coverage')


def kl_snapshot():
    rows = []
    torch.set_num_threads(1)
    for f in range(1, 6):
        v = np.load(ART/f'prepared/fold_{f:02}/val.npz')
        for seed in [42, 43, 44]:
            model = ConditionalVAE().eval()
            model.load_state_dict(torch.load(ART/f'cvae/fold_{f:02}/seed_{seed}/best_model.pt', weights_only=True, map_location='cpu'))
            total = np.zeros(16)
            with torch.inference_mode():
                for i in range(0, len(v['c']), 256):
                    mu, lv = model.encoder(torch.from_numpy(v['x'][i:i+256]), torch.from_numpy(v['c'][i:i+256]))
                    total += (.5*(mu.square()+lv.exp()-1-lv)).sum(0).numpy()
            for dim, value in enumerate(total/len(v['c'])):
                rows.append(dict(fold=f, seed=seed, latent_dimension=dim, val_kl=value,
                    scope='post-hoc validation snapshot at final checkpoint; not epoch history'))
    pd.DataFrame(rows).to_csv(OUT/'kl_per_dimension_final_validation.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-csv'); p.add_argument('--skip-cpu', action='store_true')
    p.add_argument('--skip-kl', action='store_true'); args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    if not args.motion_csv and (OUT/'motion/motion_windows.csv').exists():
        args.motion_csv = str(OUT/'motion/motion_windows.csv')
    frames = tables(args.motion_csv)
    if not args.skip_cpu: print('CPU', cpu(), flush=True)
    figures(frames)
    if not args.skip_kl: kl_snapshot()
    print('Completion tables and figures saved', flush=True)


if __name__ == '__main__': main()
