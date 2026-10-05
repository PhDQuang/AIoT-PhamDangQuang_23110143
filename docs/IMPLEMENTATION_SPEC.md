# Implementation specification

This repository implements the corrected project proposal for conditional 8-second PPG waveform generation with a 1D conditional VAE. The project proposal is the primary source of truth; the course specification supplies the G2 framing and reporting requirements.

## Dataset and labels

PPG-DaLiA subjects S1?S15 are loaded from a configurable `PPGDALIA_ROOT`. Wrist BVP/PPG is 64 Hz, ECG reference is 700 Hz, and wrist ACC is 32 Hz. ACC is inspection-only. Published/reference HR labels are preferred and are never assumed to have the PPG sampling rate. Dataset discovery accepts common `.pkl`, `.pickle`, `.mat`, `.npz`, and subject-directory layouts and fails with an actionable error when no supported layout is recognized.

## Preprocessing

Continuous PPG segments are filtered with a fourth-order Butterworth bandpass represented as SOS (`butter(N=4, Wn=[0.5, 8], btype="bandpass", fs=64, output="sos")`) and applied with `sosfiltfilt`. Windows are 512 samples with stride 128. NaN/Inf, missing sections, nearly constant windows, clipping and suspicious flat regions are recorded in the manifest; long gaps are not interpolated. Z-score statistics are fitted on valid train signal points only for each fold, with epsilon 1e-8, then reused for validation and test.

## Folds

The fixed groups are G1={S1,S2,S3}, G2={S4,S5,S6}, G3={S7,S8,S9}, G4={S10,S11,S12}, G5={S13,S14,S15}. Fold 1 tests G1, validates G2, trains G3?G5; fold 2 tests G2, validates G3, trains G1,G4,G5; fold 3 tests G3, validates G4, trains G1,G2,G5; fold 4 tests G4, validates G5, trains G1?G3; fold 5 tests G5, validates G1, trains G2?G4. Automated checks enforce pairwise subject disjointness.

## Model

The encoder receives `[B,1,512]` and condition `c=HR/60` repeated to `[B,1,512]`, concatenated to `[B,2,512]`, followed by Conv1D 2?16?32?64 (k=7,s=2,p=3), flatten 4096, Dense 4096?64, and separate 64?16 ?/logvar heads. The decoder concatenates z (16) and c (1), Dense 17?4096, reshapes to `[B,64,64]`, then ConvTranspose1D 64?32?16?1 (k=4,s=2,p=1), ReLU between hidden layers, linear output. No BatchNorm or skip path is used. Expected parameter counts are encoder 282,544, decoder 84,081, complete cVAE 366,625.

The prior is fixed N(0,I). Reconstruction uses `z=mu`; free generation samples z from N(0,I) and never sends test PPG to the encoder.

## HR proxy and losses

The frozen differentiable proxy uses Conv1D 1?16?32?64 (k=7,s=2,p=3), ReLU, global average pooling, and Dense 64?1 predicting normalized HR c. It is trained on real train PPG and selected by validation MAE. During cVAE training it is eval-mode with `requires_grad=False`, but its call remains inside autograd so gradients reach generated PPG and the decoder. `L_rec` is per-window sample MSE, `L_KL` is the diagonal Gaussian KL divided by latent dimension, and `L_HR_rec=MAE(proxy(x_hat), c)`. The base objective is `L_rec + beta*L_KL + lambda_hr*L_HR_rec`; optional prior HR loss is disabled by default (`lambda_prior=0`).

## Training protocol

PyTorch/Adam, learning rate 1e-3, batch size 128, maximum 100 epochs, beta warmup from 0 to configured beta over 10 epochs, patience 12 after warmup. Candidate pairs are (0.001,1), (0.01,0.1), (0.01,1), (0.01,5), (0.1,1), selected using validation only. Main seeds are 42,43,44; mandatory ablations default to seed 42. Every run saves a resolved configuration, history, validation metrics, checkpoint, and environment metadata.

## Evaluation

Independent HR measurement uses `scipy.signal.find_peaks`, minimum distance 20 samples, with prominence/rules selected and locked using real validation PPG against reference HR. A window is measurable only with at least three finite peaks and HR in [40,180] bpm. Report MAE, median absolute error, measurable fraction V, A3, and failures separately at 60/80/120 bpm. Reconstruction RMSE pairs each test window with its own `z=mu` reconstruction. Morphology includes rise time, 50% pulse width, rise-time/cycle ratio and beat extraction success; diversity aligns beats by pulse foot and interpolates to 128 points with per-beat shape normalization while keeping amplitude analysis separate.

## Baseline, ablations and resources

The baseline is named **Two-Gaussian waveform synthesis baseline**. Its two Gaussian pulse parameters are fit from train beats only and evaluated with the same HR targets, sample counts and detector. Mandatory ablations are lambda_hr=0 and no condition (`c=0`, lambda_hr=0), compared on identical folds and seed 42. Optional prior-HR, beta=0 and latent-size variants are configuration-only extensions.

Resource evaluation reports encoder, decoder, cVAE and proxy parameter counts, decoder state dict bytes, and CPU single-thread decoder latency (100 warm-up, 1000 timed iterations, median/P95, excluding loading/I/O/plots). Scope is FP32 inference on a documented CPU/IoT gateway; no microcontroller or INT8 claim is made.

## Expected artifacts

Each fold stores a manifest, train-fitted normalization statistics, proxy checkpoint/metrics, and each run stores `best_model.pt`, decoder weights, resolved config, history, validation/test metrics and generation tables. Publication figures go in `artifacts/figures/`. Unexecuted experiments are marked `not_run` or `not_available`.
