"""Artifact-backed experiments; tuning reads train/validation only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .config import CVAEConfig
from .data.prepare import load_prepared
from .models.cvae import ConditionalVAE
from .models.hr_proxy import HRProxy
from .training.train_cvae import train_cvae
from .evaluation.heart_rate import generation_metrics
from .evaluation.reconstruction import reconstruction_rmse
from .evaluation.morphology import pulse_features
from .evaluation.diversity import aligned_beat_diversity
from .utils.io import save_json, load_yaml

TARGETS = (60,80,120)
CANDIDATES = ((.001,1.),(.01,.1),(.01,1.),(.01,5.),(.1,1.))


def inputs(root, fold, device):
    root = Path(root)
    prepared = root/'prepared'/f'fold_{fold:02d}'
    tx,tc = load_prepared(prepared,'train')
    vx,vc = load_prepared(prepared,'val')
    proxy_dir = root/'proxy'/f'fold_{fold:02d}'
    proxy = HRProxy().to(device)
    proxy.load_state_dict(torch.load(proxy_dir/'hr_proxy.pt',map_location=device,weights_only=True))
    proxy.freeze()
    detector = json.loads((proxy_dir/'detector_config.json').read_text(encoding='utf-8'))
    return tx,tc,vx,vc,proxy,detector


def generate_fixed(model, hr, n, seed, condition_mode='hr', batch_size=128):
    # CPU generator keeps the validation latent sample set identical across devices.
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n,model.latent_dim,generator=generator)
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for i in range(0,n,batch_size):
            c = torch.full((len(z[i:i+batch_size]),),0. if condition_mode=='zero' else hr/60.,device=device)
            result.append(model.generate(c,z[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(result)


def paired_rmse(model,x,c,condition_mode='hr',batch_size=128):
    device = next(model.parameters()).device
    errors = []
    model.eval()
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            bx = torch.as_tensor(x[i:i+batch_size],device=device)
            bc = torch.as_tensor(c[i:i+batch_size],device=device)
            if condition_mode=='zero': bc = torch.zeros_like(bc)
            y,_,_ = model.reconstruct(bx,bc)
            errors.extend(reconstruction_rmse(x[i:i+batch_size],y.cpu().numpy()))
    return np.asarray(errors)


def summarize_generation(model, detector, targets=TARGETS, n=200, seed=1042, condition_mode='hr'):
    summaries = []
    for hr in targets:
        waves = generate_fixed(model,hr,n,seed,condition_mode)
        _, summary = generation_metrics(waves,hr,**detector)
        summary.update(aligned_beat_diversity(waves,**detector))
        features = pd.DataFrame([pulse_features(w,**detector) for w in waves])
        for name in features:
            summary[f'{name}_median'] = features[name].median() if features[name].notna().any() else None
        summaries.append(summary)
    return summaries


def select_candidate(rows):
    """Proposal rule: coverage, MAE, 0.1-bpm tie with RMSE, fallback A3."""
    eligible = [r for r in rows if r['all_V_ge_95'] and r['mean_mae_hr'] is not None]
    if eligible:
        lowest = min(r['mean_mae_hr'] for r in eligible)
        return min((r for r in eligible if r['mean_mae_hr']-lowest<.1),key=lambda r:r['val_rmse'])
    return max(rows,key=lambda r:(r['mean_A3'],-r['val_rmse']))


def tune_fold(root,fold,device='cpu',max_epochs=100,num_val_samples=200,include_no_hr_control=False):
    root = Path(root)
    tx,tc,vx,vc,proxy,detector = inputs(root,fold,device)
    train_hr = tc*60
    supported = [h for h in TARGETS if train_hr.min()<=h<=train_hr.max() and np.any(np.abs(train_hr-h)<10)]
    if not supported: raise ValueError('No target HR supported by this fold train data')
    rows = []
    def monitor(model):
        metrics = summarize_generation(model,detector,supported,num_val_samples)
        return {f'prior_val_{m["target_hr"]}_{key}': m[key]
                for m in metrics for key in ('V','A3','mae_hr')}
    directory = root/'tuning'/f'fold_{fold:02d}'
    def run_candidate(beta,lambda_hr,run):
        config = CVAEConfig(fold=fold,seed=42,beta=beta,lambda_hr=lambda_hr,max_epochs=max_epochs)
        print(f'Starting fold {fold}, beta={beta}, lambda_hr={lambda_hr}, seed=42',flush=True)
        model,_ = train_cvae(tx,tc,vx,vc,proxy,config,device,run,validation_monitor=monitor)
        metrics = summarize_generation(model,detector,supported,num_val_samples)
        pd.DataFrame(metrics).to_csv(run/'prior_validation_summary.csv',index=False)
        maes = [m['mae_hr'] for m in metrics]
        row = dict(beta=beta,lambda_hr=lambda_hr,all_V_ge_95=all(m['V']>=.95 for m in metrics),
                         mean_mae_hr=float(np.mean(maes)) if all(m is not None for m in maes) else None,
                         mean_A3=float(np.mean([m['A3'] for m in metrics])),
                         val_rmse=float(paired_rmse(model,vx,vc).mean()))
        print(f'Completed: {row}',flush=True)
        return row
    control = None
    if include_no_hr_control:
        # Additional diagnostic, outside the five-candidate proposal grid.
        # beta=.01 matches three candidates; all other training settings agree.
        control = run_candidate(.01,0.,directory/'no_hr_control')
        save_json(dict(**control,role='diagnostic_only_excluded_from_selection'),directory/'no_hr_control_metrics.json')
    for beta,lambda_hr in CANDIDATES:
        rows.append(run_candidate(beta,lambda_hr,directory/f'beta_{beta}_lambda_{lambda_hr}'))
    best = select_candidate(rows)
    pd.DataFrame(rows).to_csv(directory/'candidate_comparison.csv',index=False)
    if control is not None:
        comparison = [dict(**row,role='proposal_candidate') for row in rows]
        comparison.append(dict(**control,role='diagnostic_only_excluded_from_selection'))
        pd.DataFrame(comparison).to_csv(directory/'comparison_with_no_hr_control.csv',index=False)
    save_json(dict(fold=fold,beta=best['beta'],lambda_hr=best['lambda_hr'],
                   supported_targets=supported,validation_seed=1042,num_val_samples=num_val_samples,
                   selection_rule='V>=95% at supported targets, MAE, 0.1 bpm tie RMSE; fallback A3',
                   selected_metrics=best),directory/'selected_config.json')
    return best


def train_selected(root,fold,seed,device='cpu',ablation=None):
    root = Path(root)
    selected = json.loads((root/'tuning'/f'fold_{fold:02d}'/'selected_config.json').read_text(encoding='utf-8'))
    tx,tc,vx,vc,proxy,detector = inputs(root,fold,device)
    config = CVAEConfig(fold=fold,seed=seed,beta=selected['beta'],lambda_hr=selected['lambda_hr'])
    if ablation:
        if seed!=42 or ablation not in {'no_hr_loss','no_condition'}: raise ValueError('Mandatory ablations use seed 42')
        config.lambda_hr = 0.
        if ablation=='no_condition': config.condition_mode = 'zero'
    run = (root/'ablations'/ablation if ablation else root/'cvae')/f'fold_{fold:02d}'/f'seed_{seed}'
    model,history = train_cvae(tx,tc,vx,vc,proxy,config,device,run)
    pd.DataFrame(summarize_generation(model,detector,selected['supported_targets'],condition_mode=config.condition_mode)).to_csv(run/'prior_validation_summary.csv',index=False)
    return run,history


def load_run(run,device='cpu'):
    run = Path(run)
    config = load_yaml(run/'resolved_config.yaml')
    model = ConditionalVAE(config['latent_dim']).to(device)
    model.load_state_dict(torch.load(run/'best_model.pt',map_location=device,weights_only=True))
    model.eval()
    return model,config


def fit_baseline(root,fold,seed=42,max_train_windows=500):
    from dataclasses import asdict
    from .evaluation.beats import extract_beats
    from .baselines.gaussian import fit_two_gaussian
    root = Path(root)
    tx,_ = load_prepared(root/'prepared'/f'fold_{fold:02d}','train')
    detector = json.loads((root/'proxy'/f'fold_{fold:02d}'/'detector_config.json').read_text(encoding='utf-8'))
    indices = np.random.default_rng(42).choice(len(tx),min(max_train_windows,len(tx)),replace=False)
    beats = []
    for i in indices:
        extracted,_ = extract_beats(tx[i],**detector)
        beats.extend(extracted[:1])
    bank = fit_two_gaussian(beats)
    run = root/'gaussian'/f'fold_{fold:02d}'/f'seed_{seed}'
    save_json({'fit_split':'train','fit_seed':42,'max_train_windows':max_train_windows,
               'num_train_beats':len(beats),'num_fitted_beats':len(bank),
               'parameters':[asdict(p) for p in bank]},run/'gaussian_fit.json')
    return run


def _wave_tables(waves,hr,detector,metadata):
    rows,summary = generation_metrics(waves,hr,**detector)
    morphology = []
    for i,w in enumerate(waves):
        morphology.append(dict(index=i,target_hr=hr,**metadata,**pulse_features(w,**detector),
                               amplitude_range=float(np.ptp(w)), amplitude_std=float(np.std(w))))
    for row in rows: row.update(metadata)
    summary.update(metadata)
    summary.update(aligned_beat_diversity(waves,**detector))
    return rows,summary,morphology


def _plot_samples(waves,rows,hr,run):
    import matplotlib.pyplot as plt
    t = np.arange(512)/64
    # Fixed first 20 indices, independent of measured HR and model quality.
    fig,axes = plt.subplots(5,4,figsize=(16,10),sharex=True)
    for i,ax in enumerate(axes.flat):
        if i<len(waves):
            ax.plot(t,waves[i].reshape(-1),lw=.8)
            ax.set_title(f'index {i}, measurable={rows[i]["measurable"]}')
        else: ax.set_visible(False)
    fig.suptitle(f'HR target {hr} bpm, fixed indices 0..19')
    fig.tight_layout()
    fig.savefig(run/f'generated_{hr}_fixed_samples.png',dpi=150)
    plt.close(fig)
    failed = [r['index'] for r in rows if not r['measurable']][:5]
    if failed:
        fig,axes = plt.subplots(len(failed),1,figsize=(12,2*len(failed)),squeeze=False)
        for i,ax in zip(failed,axes.flat):
            ax.plot(t,waves[i].reshape(-1))
            ax.set_title(f'Failure index {i}: {rows[i]["failure_reason"]}')
        fig.tight_layout()
        fig.savefig(run/f'generated_{hr}_failures.png',dpi=150)
        plt.close(fig)


def evaluate_run(root,fold,seed,method='cvae',device='cpu',num_samples=1000):
    """Call only after locking every fold's protocol; opens real test data."""
    root = Path(root)
    if method in {'no_hr_loss','no_condition'}:
        run = root/'ablations'/method/f'fold_{fold:02d}'/f'seed_{seed}'
    elif method in {'cvae','gaussian'}:
        run = root/method/f'fold_{fold:02d}'/f'seed_{seed}'
    else: raise ValueError('Unknown method')
    detector = json.loads((root/'proxy'/f'fold_{fold:02d}'/'detector_config.json').read_text(encoding='utf-8'))
    metadata = dict(method=method,fold=fold,seed=seed)
    if method=='gaussian':
        from .baselines.gaussian import GaussianParams,synthesize_two_gaussian
        fitted = json.loads((run/'gaussian_fit.json').read_text(encoding='utf-8'))
        bank = [GaussianParams(**p) for p in fitted['parameters']]
        generate = lambda hr: synthesize_two_gaussian(hr,num_samples,params=bank,seed=seed)
    else:
        model,config = load_run(run,device)
        generate = lambda hr: generate_fixed(model,hr,num_samples,seed,config['condition_mode'])
    all_rows, summaries, morphology = [],[],[]
    for hr in TARGETS:
        waves = generate(hr)
        np.savez_compressed(run/f'generated_{hr}.npz',ppg=waves,target_hr=hr,seed=seed)
        rows,summary,features = _wave_tables(waves,hr,detector,metadata)
        all_rows.extend(rows)
        summaries.append(summary)
        morphology.extend(features)
        _plot_samples(waves,rows,hr,run)
    pd.DataFrame(all_rows).to_csv(run/'generation_samples.csv',index=False)
    pd.DataFrame(summaries).to_csv(run/'generation_summary.csv',index=False)
    pd.DataFrame(morphology).to_csv(run/'generated_morphology.csv',index=False)
    prepared = root/'prepared'/f'fold_{fold:02d}'
    test_x,test_c = load_prepared(prepared,'test')
    manifest = pd.read_csv(prepared/'manifest.csv')
    test_rows = manifest[(manifest.split=='test') & manifest.valid].reset_index(drop=True)
    if method!='gaussian':
        test_rows['rmse'] = paired_rmse(model,test_x,test_c,config['condition_mode'])
        test_rows.to_csv(run/'paired_test_reconstruction.csv',index=False)
        test_rows.groupby('subject_id').rmse.agg(['mean','median','count']).to_csv(run/'reconstruction_by_subject.csv')
    # Balance each HR-bin reference by subject: equal counts, maximum 100 each.
    rng = np.random.default_rng(2042)
    reference = []
    reference_summary = []
    for hr in TARGETS:
        pool = test_rows[test_rows.hr_reference_bpm.between(hr-10,hr+10,inclusive='left')]
        groups = [group.index.to_numpy() for _,group in pool.groupby('subject_id')]
        count = min([100]+[len(g) for g in groups]) if groups else 0
        indices = np.concatenate([rng.choice(g,count,replace=False) for g in groups]) if groups else np.array([],dtype=int)
        for i in indices:
            reference.append(dict(index=int(i),target_hr=hr,subject_id=test_rows.loc[i,'subject_id'],
                                  **pulse_features(test_x[i],**detector),
                                  amplitude_range=float(np.ptp(test_x[i])), amplitude_std=float(np.std(test_x[i]))))
        diversity = aligned_beat_diversity(test_x[indices],**detector) if count else {'mean_temporal_std':None,'num_beats':0}
        reference_summary.append(dict(target_hr=hr,windows_per_present_subject=count,
                                      subjects_present=len(groups),subjects_total=test_rows.subject_id.nunique(),
                                      status='available' if count else 'insufficient_reference',**diversity))
    pd.DataFrame(reference).to_csv(run/'balanced_test_morphology.csv',index=False)
    pd.DataFrame(reference_summary).to_csv(run/'test_reference_coverage.csv',index=False)
    generated = pd.DataFrame(morphology)
    real = pd.DataFrame(reference)
    comparison = []
    names = ['rise_time_s','width50_s','rise_cycle_ratio','beat_extraction_success','amplitude_range','amplitude_std']
    for source,frame in [('generated',generated),('real_test',real)]:
        if frame.empty: continue
        for hr,group in frame.groupby('target_hr'):
            for name in names:
                values = group[name].dropna()
                comparison.append(dict(source=source,target_hr=hr,feature=name,count=len(values),
                                       median=values.median() if len(values) else None,
                                       q25=values.quantile(.25) if len(values) else None,
                                       q75=values.quantile(.75) if len(values) else None))
    pd.DataFrame(comparison).to_csv(run/'morphology_comparison.csv',index=False)
    save_json(dict(**metadata,num_samples=num_samples,final_latent_seed=seed,
                   validation_latent_seed=1042,detector=detector),run/'evaluation_config.json')
    return pd.DataFrame(summaries)
