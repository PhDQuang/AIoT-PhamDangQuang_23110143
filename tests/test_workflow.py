import json
import numpy as np
import pandas as pd
import torch
from ppg_cvae.data.prepare import prepare_fold, load_prepared
from ppg_cvae.training.train_hr_proxy import train_proxy
from ppg_cvae.evaluation.calibration import calibrate_detector
from ppg_cvae.evaluation.aggregate import summarize_runs
from ppg_cvae.baselines.gaussian import fit_two_gaussian, synthesize_two_gaussian, GaussianParams, _pulse
from ppg_cvae import workflow


def test_candidate_selection_uses_coverage_mae_tie_and_fallback():
    a=dict(beta=.01,lambda_hr=1.,all_V_ge_95=True,mean_mae_hr=2.,mean_A3=.6,val_rmse=1.)
    b=dict(beta=.1,lambda_hr=1.,all_V_ge_95=True,mean_mae_hr=2.05,mean_A3=.7,val_rmse=.5)
    c=dict(beta=.001,lambda_hr=1.,all_V_ge_95=False,mean_mae_hr=.1,mean_A3=.8,val_rmse=.1)
    assert workflow.select_candidate([a,b,c]) == b
    assert workflow.select_candidate([{**a,'all_V_ge_95':False},c]) == c


def test_gaussian_fits_shape_and_amplitude_from_real_beats():
    true=GaussianParams(2.,.3,.06,.6,.65,.09,-.2)
    phase=np.linspace(0,1,128)
    beat=_pulse(phase,true)
    fitted=fit_two_gaussian([beat])
    assert np.sqrt(np.mean((_pulse(phase,fitted[0])-beat)**2)) < .01
    assert abs(fitted[0].mu1-.3) < .01
    assert synthesize_two_gaussian(80,3,params=fitted,seed=42).shape == (3,1,512)


def test_aggregation_averages_seeds_before_folds():
    rows=pd.DataFrame([dict(method='cvae',fold=1,seed=s,target_hr=60,mae_hr=v) for s,v in [(42,1.),(43,3.),(44,5.)]]+
                      [dict(method='cvae',fold=2,seed=42,target_hr=60,mae_hr=9.)])
    folds,summary=summarize_runs(rows)
    assert folds.mae_hr_mean.tolist() == [3.,9.]
    assert summary.mae_hr_mean_mean.iloc[0] == 6.


def test_synthetic_artifact_pipeline(tmp_path,monkeypatch):
    # Reduced budgets validate wiring and artifact contracts, not real model quality.
    torch.set_num_threads(1)
    raw=tmp_path/'raw'; raw.mkdir()
    t=np.arange(2048)/64
    for i in range(1,16):
        np.savez(raw/f'S{i}.npz',ppg=np.sin(2*np.pi*t),hr=np.full(13,60.))
    root=tmp_path/'artifacts'; prepared=root/'prepared'/'fold_01'
    prepare_fold(raw,prepared,label_start_sample=0,alignment_source='synthetic protocol')
    tx,tc=load_prepared(prepared,'train'); vx,vc=load_prepared(prepared,'val')
    proxy_dir=root/'proxy'/'fold_01'
    train_proxy(tx,tc,vx,vc,epochs=1,output_dir=proxy_dir)
    calibrate_detector(vx,vc,proxy_dir)
    best=workflow.tune_fold(root,1,max_epochs=1,num_val_samples=4,include_no_hr_control=True)
    tuning_dir=root/'tuning'/'fold_01'
    comparison=pd.read_csv(tuning_dir/'comparison_with_no_hr_control.csv')
    assert len(pd.read_csv(tuning_dir/'candidate_comparison.csv'))==5
    assert len(comparison)==6
    assert comparison[comparison.lambda_hr==0].role.eq('diagnostic_only_excluded_from_selection').all()
    assert best['lambda_hr']>0
    selected=json.loads((root/'tuning'/'fold_01'/'selected_config.json').read_text())
    assert best['beta'] == selected['beta']
    original_train=workflow.train_cvae
    def short_train(*args,**kwargs):
        args[5].max_epochs=1
        return original_train(*args,**kwargs)
    monkeypatch.setattr(workflow,'train_cvae',short_train)
    run,_=workflow.train_selected(root,1,42)
    assert (run/'best_model.pt').exists()
    # Skip graphics here; numeric evaluation and reconstruction must execute.
    monkeypatch.setattr(workflow,'_plot_samples',lambda *args:None)
    summary=workflow.evaluate_run(root,1,42,num_samples=4)
    assert summary.num_samples.tolist() == [4,4,4]
    assert (run/'paired_test_reconstruction.csv').exists()
    baseline=workflow.fit_baseline(root,1,max_train_windows=3)
    assert json.loads((baseline/'gaussian_fit.json').read_text())['fit_split']=='train'
    result=workflow.evaluate_run(root,1,42,'gaussian',num_samples=4)
    assert result.method.eq('gaussian').all()
