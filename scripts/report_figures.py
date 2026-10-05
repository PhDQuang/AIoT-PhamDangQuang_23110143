"""Read experiment artifacts and prepare grayscale scientific report figures."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch
from ppg_cvae.models.cvae import ConditionalVAE

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'
OUT=ROOT/'reports/work/figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':220})
def save(name):
    plt.savefig(OUT/f'{name}.png',bbox_inches='tight',facecolor='white'); plt.close()
def box(ax,x,y,w,h,t):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='0.96',edgecolor='black',lw=1))
    ax.text(x+w/2,y+h/2,t.replace('⊙','×'),ha='center',va='center',fontsize=10)
def arrow(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='black'))

fig,ax=plt.subplots(figsize=(9,3.7)); ax.set(xlim=(0,10),ylim=(0,4)); ax.axis('off')
for x,t in [(0.1,'PPG DaLiA\n15 người'),(2.65,'Chia người\nTrain / Val / Test'),(5.2,'Lọc và cắt\n512 mẫu / bước 128'),(7.75,'Chuẩn hóa\nThống kê train')]: box(ax,x,2.5,2.05,1,t)
for x in [.1,2.65,5.2]: arrow(ax,(x+2.05,3),(x+2.55,3))
for x,t in [(7.75,'Proxy và tuning\nChỉ dùng train / val'),(5.2,'Khóa protocol\n5 fold và 3 seed'),(2.65,'Sinh từ prior\n60 / 80 / 120 bpm'),(.1,'Đánh giá test\nHR / hình thái / CPU')]: box(ax,x,.5,2.05,1,t)
arrow(ax,(8.775,2.5),(8.775,1.5))
for x in [7.75,5.2,2.65]: arrow(ax,(x,1),(x-.5,1))
save('pipeline')

fig,ax=plt.subplots(figsize=(8.3,3.5)); mat=np.zeros((5,5),int)
for f in range(5): mat[f,f]=2; mat[f,(f+1)%5]=1
ax.imshow(mat,cmap=matplotlib.colors.ListedColormap(['white','0.75','0.35']),vmin=0,vmax=2,aspect='auto')
for i in range(5):
 for j in range(5): ax.text(j,i,['Train','Val','Test'][mat[i,j]],ha='center',va='center',color='white' if mat[i,j]==2 else 'black')
ax.set_xticks(range(5),[f'G{i+1}\nS{3*i+1}–S{3*i+3}' for i in range(5)]); ax.set_yticks(range(5),[f'Fold {i+1}' for i in range(5)]); save('folds')

summary=pd.read_csv(A/'prepared/fold_01/subject_summary.csv')
manifest=pd.read_csv(A/'prepared/fold_01/manifest.csv'); valid=manifest[manifest.valid]
fig,axs=plt.subplots(1,2,figsize=(9,3.5))
for split,ls in [('train','-'),('val','--'),('test',':')]:
 v=valid.loc[valid.split==split,'hr_reference_bpm']; hist,b=np.histogram(v,bins=np.arange(40,201,10)); axs[0].plot((b[:-1]+b[1:])/2,hist,ls,color={'train':'black','val':'.4','test':'.65'}[split],label=split)
axs[0].set(xlabel='HR tham chiếu (bpm)',ylabel='Số cửa sổ',title='Phân bố HR tại fold 1'); axs[0].legend()
cov=pd.read_csv(A/'prepared/fold_01/hr_coverage.csv'); colors=['.2','.5','.8']
for i,split in enumerate(['train','val','test']): axs[1].bar(np.arange(3)+(i-1)*.25,cov[cov.split==split].windows,width=.25,color=colors[i],label=split,edgecolor='black',lw=.4)
axs[1].set(xticks=np.arange(3),xticklabels=['60','80','120'],xlabel='HR mục tiêu ±10 bpm',ylabel='Số cửa sổ',title='Độ phủ dữ liệu tại fold 1'); axs[1].legend(); fig.tight_layout(); save('coverage')

fig,ax=plt.subplots(figsize=(9,4.6)); ax.set(xlim=(0,10),ylim=(0,5)); ax.axis('off')
boxes=[(.1,3.5,2.1,1,'x và c lặp\n[B,2,512]'),(2.7,3.5,3,1,'Conv1D 2→16→32→64\nk7 s2 p3; [B,64,64]'),(6.2,3.5,3.5,1,'Flatten 4096 → Dense 64\nμ và log σ²: [B,16]'),(6.2,1.8,3.5,1,'z = μ + σ ⊙ ε\nSinh tự do: z ~ N(0,I)'),(6.2,.1,3.5,1,'Ghép z và c: [B,17]\nDense 4096 → [B,64,64]'),(2.7,.1,3,1,'ConvTranspose1D\n64→32→16→1; k4 s2 p1'),(.1,.1,2.1,1,'PPG đầu ra\n[B,1,512]')]
for b in boxes: box(ax,*b)
for a,b in [((2.2,4),(2.7,4)),((5.7,4),(6.2,4)),((7.95,3.5),(7.95,2.8)),((7.95,1.8),(7.95,1.1)),((6.2,.6),(5.7,.6)),((2.7,.6),(2.2,.6))]: arrow(ax,a,b)
ax.text(3.4,2.3,'Encoder chỉ dùng khi học và tái tạo\nDecoder dùng khi sinh từ prior\nĐiều kiện c = HR / 60',ha='center',va='center',fontsize=11)
save('architecture')

quality=[json.loads((A/f'tuning/fold_{f:02}/measurement_quality.json').read_text()) for f in range(1,6)]
fig,ax=plt.subplots(figsize=(8.5,3.2)); x=np.arange(5)
ax.bar(x-.18,[q['proxy']['best_val_mae_bpm'] for q in quality],.36,color='.3',label='Proxy khả vi'); ax.bar(x+.18,[q['detector']['selected']['mae_hr'] for q in quality],.36,color='.8',edgecolor='black',label='Detector độc lập')
ax.set(xticks=x,xticklabels=[f'Fold {i}' for i in range(1,6)],ylabel='MAE trên PPG thật val (bpm)',ylim=(0,14)); ax.legend(ncol=2); save('proxy')

h=pd.read_csv(A/'cvae/fold_01/seed_42/training_history.csv')
fig,axs=plt.subplots(1,3,figsize=(9,3.1))
for ax,k,t in zip(axs,['L_rec','L_KL','L_HR_rec'],['MSE tái tạo','KL chia 16','HR loss đơn vị c']):
 ax.plot(h.epoch,h['train_'+k],color='black',label='Train'); ax.plot(h.epoch,h['val_'+k],color='.5',ls='--',label='Val'); ax.set(title=t,xlabel='Epoch'); ax.legend(fontsize=8)
fig.tight_layout(); save('training')

allruns=pd.read_csv(A/'summary/all_runs.csv'); cross=pd.read_csv(A/'summary/cross_fold_summary.csv')
for metric,label,name,mult in [('mae_hr','MAE HR (bpm)','mae',1),('A3','Cửa sổ có sai số ≤3 bpm (%)','a3',100)]:
 fig,ax=plt.subplots(figsize=(8.5,3.4)); x=np.arange(3)
 for i,m in enumerate(['cvae','gaussian']):
  z=cross[cross.method==m].sort_values('target_hr'); ax.bar(x+(i-.5)*.34,z[f'{metric}_mean_mean']*mult,.34,yerr=z[f'{metric}_mean_std']*mult,capsize=4,color=['.3','.8'][i],edgecolor='black',label=['cVAE','Hai Gaussian'][i])
 ax.set(xticks=x,xticklabels=['60','80','120'],xlabel='HR mục tiêu (bpm)',ylabel=label); ax.legend(); ax.set_ylim(0,110 if mult==100 else 44); save(name)
fig,ax=plt.subplots(figsize=(8.5,3.2));
for i,m in enumerate(['cvae','gaussian']):
 z=cross[cross.method==m].sort_values('target_hr'); ax.plot(z.target_hr,100*z.V_mean_mean,['o-','s--'][i],color=['black','.5'][i],label=['cVAE','Hai Gaussian'][i])
ax.set(ylim=(0,105),xticks=[60,80,120],xlabel='HR mục tiêu (bpm)',ylabel='Tỷ lệ đo được V (%)'); ax.legend(); save('validity')

samples=pd.read_csv(A/'cvae/fold_01/seed_42/generation_samples.csv')
fig,axs=plt.subplots(1,3,figsize=(9,3.2))
for ax,hr in zip(axs,[60,80,120]):
 s=samples[samples.target_hr==hr]; ax.hist(s.measured_hr,bins=np.arange(40,185,5),color='.5',edgecolor='white'); ax.axvline(hr,color='black',ls='--',lw=2); ax.set(title=f'Mục tiêu {hr} bpm',xlabel='HR đo được (bpm)',ylabel='Số cửa sổ',xlim=(40,180))
fig.tight_layout(); save('hr_distribution')

fig,axs=plt.subplots(3,2,figsize=(9,5.2),sharex=True); t=np.arange(512)/64
for row,hr in enumerate([60,80,120]):
 for col,m in enumerate(['cvae','gaussian']):
  z=np.load(A/f'{m}/fold_01/seed_42/generated_{hr}.npz')['ppg'][0,0]; s=pd.read_csv(A/f'{m}/fold_01/seed_42/generation_samples.csv'); mh=s[(s.target_hr==hr)&(s['index']==0)].measured_hr.iloc[0]
  axs[row,col].plot(t,z,color='black',lw=.9); axs[row,col].set_title(f'{["cVAE","Hai Gaussian"][col]} | h={hr} | đo={mh:.1f}',fontsize=10); axs[row,col].set_ylabel('Biên độ Z')
for ax in axs[-1]: ax.set_xlabel('Thời gian (s)')
fig.tight_layout(); save('waveforms')

torch.set_num_threads(1); ck=torch.load(A/'cvae/fold_01/seed_42/best_model.pt',map_location='cpu',weights_only=False)
print('checkpoint keys',list(ck))
model=ConditionalVAE(); model.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck); model.eval()
test=np.load(A/'prepared/fold_01/test.npz'); print('test keys',test.files)
tx=test['x'][:3]; tc=test['c'][:3]
with torch.no_grad(): rec=model.reconstruct(torch.as_tensor(tx),torch.as_tensor(tc))[0].numpy()
fig,axs=plt.subplots(3,1,figsize=(9,4.5),sharex=True)
for i,ax in enumerate(axs):
 ax.plot(t,tx[i,0],color='black',lw=1,label='PPG test thật'); ax.plot(t,rec[i,0],color='.55',lw=1,ls='--',label='Tái tạo z=μ'); ax.set_ylabel('Biên độ Z'); ax.set_title(f'Cửa sổ test thứ {i+1} | RMSE={np.sqrt(np.mean((tx[i]-rec[i])**2)):.3f}',fontsize=9)
axs[0].legend(ncol=2,fontsize=8); axs[-1].set_xlabel('Thời gian (s)'); fig.tight_layout(); save('reconstruction')

fig,axs=plt.subplots(1,3,figsize=(9,3.2)); recon=[]
for m,path in [('cvae','cvae'),('no_hr_loss','ablations/no_hr_loss'),('no_condition','ablations/no_condition')]:
 for f in range(1,6):
  seeds=[42,43,44] if m=='cvae' else [42]
  vals=[pd.read_csv(A/f'{path}/fold_{f:02}/seed_{s}/paired_test_reconstruction.csv').rmse.mean() for s in seeds]; recon.append({'method':m,'fold':f,'rmse':float(np.mean(vals))})
for ax,m in zip(axs,['cvae','no_hr_loss','no_condition']):
 rows=[r for r in recon if r['method']==m]; ax.bar(range(1,6),[r['rmse'] for r in rows],color='.5'); ax.set(title=m,xlabel='Fold',ylabel='RMSE trung bình',ylim=(0,max(r['rmse'] for r in recon)*1.15))
fig.tight_layout(); save('rmse')

fig,axs=plt.subplots(1,3,figsize=(9,3.4)); morphology=pd.read_csv(A/'cvae/fold_01/seed_42/morphology_comparison.csv'); gauss=pd.read_csv(A/'gaussian/fold_01/seed_42/morphology_comparison.csv')
for ax,feature,title in zip(axs,['rise_time_s','width50_s','amplitude_std'],['Thời gian lên đỉnh (s)','Độ rộng xung 50% (s)','Độ lệch chuẩn biên độ Z']):
 for i,(df,src,label,ls) in enumerate([(morphology,'real_test','PPG thật','o-'),(morphology,'generated','cVAE','s--'),(gauss,'generated','Hai Gaussian','^:')]):
  z=df[(df.feature==feature)&(df.source==src)].sort_values('target_hr'); ax.errorbar(np.arange(3)+(i-1)*.09,z['median'],yerr=np.array([z['median']-z.q25,z.q75-z['median']]),fmt=ls,color=['black','.4','.7'][i],capsize=3,label=label)
 ax.set(xticks=range(3),xticklabels=['60','80','120'],title=title,xlabel='HR mục tiêu (bpm)')
axs[0].legend(fontsize=8); fig.tight_layout(); save('morphology')

fig,ax=plt.subplots(figsize=(8.5,3.2))
for m,ls,color,label in [('cvae','o-','black','cVAE'),('gaussian','s--','.45','Hai Gaussian')]:
 z=cross[cross.method==m].sort_values('target_hr'); ax.plot(z.target_hr,z.mean_temporal_std_mean_mean,ls,color=color,label=label)
realdiv=[]
for hr in [60,80,120]: realdiv.append(np.mean([pd.read_csv(A/f'cvae/fold_{f:02}/seed_42/test_reference_coverage.csv').set_index('target_hr').loc[hr,'mean_temporal_std'] for f in range(1,6)]))
ax.plot([60,80,120],realdiv,'^:',color='.7',label='PPG thật tham chiếu'); ax.set(xticks=[60,80,120],ylim=(0,.3),xlabel='HR mục tiêu (bpm)',ylabel='Phân tán nhịp sau chuẩn hóa hình dạng'); ax.legend(); save('diversity')

fair=allruns[allruns.seed==42].groupby(['method','target_hr']).mae_hr.agg(['mean','std']).reset_index()
fig,ax=plt.subplots(figsize=(8.5,3.4));
for i,m in enumerate(['cvae','no_hr_loss','no_condition']):
 z=fair[fair.method==m].sort_values('target_hr'); ax.bar(np.arange(3)+(i-1)*.25,z['mean'],.25,yerr=z['std'],capsize=3,color=['.2','.55','.85'][i],edgecolor='black',label=m)
ax.set(xticks=range(3),xticklabels=['60','80','120'],xlabel='HR mục tiêu (bpm)',ylabel='MAE HR cùng seed 42 (bpm)'); ax.legend(); save('ablation')
resources=pd.read_csv(A/'summary/resource_metrics.csv')
fig,axs=plt.subplots(1,2,figsize=(9,3.2))
for seed,ls in [(42,'o-'),(43,'s--'),(44,'^:')]:
 z=resources[resources.seed==seed].sort_values('fold'); axs[0].plot(z.fold,z.decoder_latency_ms_median,ls,color={42:'black',43:'.4',44:'.7'}[seed],label=f'Seed {seed}'); axs[1].plot(z.fold,z.decoder_latency_ms_p95,ls,color={42:'black',43:'.4',44:'.7'}[seed])
for ax,title in zip(axs,['Median','P95']): ax.set(xticks=range(1,6),xlabel='Fold',ylabel='Thời gian decoder (ms)',title=title,ylim=(0,.45))
axs[0].legend(fontsize=8); fig.tight_layout(); save('latency')

fig,ax=plt.subplots(figsize=(9,3.3)); ax.set(xlim=(0,10),ylim=(0,3)); ax.axis('off')
for x,t in [(.1,'HR mục tiêu\nvà seed z'),(2.65,'Decoder FP32\nCPU gateway'),(5.2,'PPG số\n512 mẫu / 8 giây'),(7.75,'Thuật toán IoT\nKiểm thử đầu vào')]: box(ax,x,1.25,2.05,1.2,t)
for x in [.1,2.65,5.2]: arrow(ax,(x+2.05,1.85),(x+2.55,1.85))
ax.text(5,.45,'Sơ đồ tích hợp đề xuất    |    Chưa triển khai gateway hoặc MCU thực tế',ha='center',fontsize=11); save('iot')
metrics={'dataset':[], 'reconstruction':recon,'fair_ablation':fair.to_dict('records'),'proxy':[{'fold':i+1,'proxy':q['proxy']['best_val_mae_bpm'],'detector':q['detector']['selected']['mae_hr']} for i,q in enumerate(quality)],'real_diversity':realdiv}
for f in range(1,6):
 m=pd.read_csv(A/f'prepared/fold_{f:02}/manifest.csv'); metrics['dataset'].append({'fold':f,'splits':m.groupby('split').agg(total=('valid','size'),valid=('valid','sum')).reset_index().to_dict('records')})
(OUT.parent/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
print('FIGURES',len(list(OUT.glob('*.png')))); print(json.dumps(metrics,ensure_ascii=False))
