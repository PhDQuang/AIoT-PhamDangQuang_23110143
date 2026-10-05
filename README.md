# Sinh PPG có điều kiện theo HR bằng 1D cVAE

Sinh viên: **Phạm Đăng Quang – 23110143**, đề tài G2, học phần Trí tuệ nhân tạo cho IoT.

Repository: https://github.com/PhDQuang/AIoT-PhamDangQuang_23110143

## Chạy demo từ repository

Mở PowerShell tại thư mục project, dùng Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.\scripts\start_web.ps1
```

Mở http://127.0.0.1:8000. Checkpoint trong `results/cvae/fold_01/seed_42/` và ngân hàng Gaussian cùng fold được cung cấp sẵn, không cần tải dataset để chạy demo.

## Cấu trúc và hồ sơ cuối kỳ

| Thư mục | Nội dung |
|---|---|
| `src/ppg_cvae/` | Tiền xử lý, mô hình, huấn luyện, đánh giá và API web |
| `notebooks/` | Notebook nguồn và [bản đã thực thi](notebooks/README.md) |
| `configs/` | Cấu hình dữ liệu và thí nghiệm |
| `results/` | Checkpoint, normalization, detector và kết quả theo fold |
| `completion/` | Thống kê bổ sung và phép đo CPU máy thử |
| `web/` | Giao diện và hướng dẫn chạy demo |
| `reports/` | [Word có bìa](reports/PhamDangQuang_final.docx), [PowerPoint](reports/PhamDangQuang_final.pptx), hình và nội dung slide |

Tên hồ sơ nộp bài: `PhamDangQuang_final`. Gói mã nguồn nén được nộp riêng trên LMS cùng Word và PPTX, xem [hướng dẫn nộp bài](HUONG_DAN_NOP_BAI.md).

Dữ liệu cảm biến thô, NPZ huấn luyện/sinh lớn, cache, output job gốc và môi trường Python được giữ tại máy làm việc, nằm ngoài repository. Kết quả đã thực thi giữ thông tin job nguồn; notebook nguồn chưa có output thực thi vẫn được giữ để chạy lại.

Kiểm tra mã nguồn:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/validate_repo.py
```

## Quy trình nghiên cứu

Project sinh PPG 8 giây, 512 mẫu ở 64 Hz, với điều kiện HR 60/80/120 bpm. Kiến trúc đã chạy trên PPG-DaLiA thật: 282.544 tham số encoder, 84.081 decoder, 366.625 cVAE và 18.209 HR proxy.

Kết quả cuối thuộc job `20261004-151954-483f31`: đủ 5 fold, cVAE/Gaussian ba seed và hai ablation seed 42, tổng 120.000 cửa sổ sinh. Xem [kiểm tra kết quả](docs/FINAL_RESULTS_REVIEW_2026-10-05.md) và [đối chiếu yêu cầu](docs/REQUIREMENTS_AUDIT_2026-10-05.md). Các review trước ngày 05/10 là lịch sử xử lý lỗi, không phải trạng thái hiện tại.

## Cài đặt local

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/validate_repo.py
```

Có thể dùng `pip install -e .` để import mà không cần PYTHONPATH. Đặt `PPGDALIA_ROOT` tới thư mục chứa duy nhất một bộ S1–S15. Loader hỗ trợ pickle, NPZ với khóa PPG/HR trực tiếp và MAT chứa struct. Kết quả tải về ở `.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts/`; gói bàn giao dùng thư mục `results/`.

## Quy trình chạy

1. Chạy lại `00_kaggle_setup_and_data_audit` và `00_eda_ppg_dalia`. Kiểm tra đủ 15 mã người duy nhất. Đường HR ở biểu đồ EDA hiện minh họa với giả định s0=0.
2. Đọc readme đi kèm dataset để xác nhận mốc nhãn. Trong notebook 01 khai báo `LABEL_START_SAMPLE` và `ALIGNMENT_SOURCE`. Mã dừng nếu chưa khai báo hoặc số nhãn không khớp số cửa sổ.
3. `01_build_manifest_and_preprocess`: lưu manifest, normalization chỉ từ điểm train hợp lệ không đếm lặp, train/val/test NPZ, thống kê người và độ phủ HR. Lọc từng đoạn hữu hạn riêng, không nối qua khoảng thiếu. Margin biên lọc mặc định 128 mẫu.
4. `02_train_hr_proxy`: train thật, chọn checkpoint bằng validation MAE, lưu sai số bpm theo người/dải HR và chọn detector trên PPG thật validation. Xem các bảng trước khi quyết định dùng loss HR.
5. `03_tune_cvae`: năm cặp beta/lambda, seed 42, theo dõi prior validation với z cố định. Chọn bằng V/MAE/A3 và RMSE theo đề cương, không so loss tổng giữa các trọng số khác nhau.
6. `04_train_final_cross_validation`: chạy fold 1–5 và seed 42/43/44 với cấu hình đã chọn. `05_train_evaluate_gaussian_baseline` khớp ngân hàng tham số trên nhịp train; `06_run_ablations` train hai đối chứng seed 42.
7. `07_final_evaluation_and_figures`: đặt `EVALUATION_PROTOCOL_LOCKED=True` sau khi khóa quy trình mọi fold. Sinh 1.000 mẫu/HR/phương pháp; lưu mẫu lỗi, RMSE có cặp, hình thái/đa dạng và tham chiếu thật cân bằng theo người trong từng dải HR. Hình minh họa dùng 20 chỉ số cố định.
8. `08_export_decoder_and_cpu_benchmark`: nạp checkpoint thật, đo CPU FP32 batch 1 một luồng, 100 warm-up/1.000 lần đo, byte tệp và metadata phần cứng.
9. `python scripts/aggregate_results.py artifacts`: trung bình seed trong fold rồi tổng hợp mean/std giữa fold. Kiểm tra số run/fold thực tế, không coi kết quả thiếu là đủ 5 fold.

Mọi mô hình trong một fold dùng chung manifest, normalization, proxy và detector. Hai ablation dùng beta đã chọn và seed 42. Kết quả chưa chạy phải ghi `not_run`.

## Web demo tương tác

Chạy `.\scripts\start_web.ps1`, sau đó mở http://127.0.0.1:8000. Chọn HR 60/80/120 bpm và seed để sinh cửa sổ PPG 8 giây bằng decoder thật, đối chiếu hai Gaussian và xem HR đo độc lập, MAE, V, A3. Có xuất PNG/CSV/JSON. Xem [hướng dẫn web](web/README.md).

cVAE sinh PPG theo HR và latent từ prior. Mỗi cửa sổ là độc lập; giao diện chính trình diễn đúng nhiệm vụ tạo sinh G2, không dùng file lịch sử để dự báo tương lai.

## Artifact giữa các session Kaggle

Mặc định `OUTPUT_ROOT=Path('artifacts')` để KRun thu đúng đường dẫn trong `krun.yaml`. Nếu đổi `PPGCVAE_OUTPUT_ROOT` ra ngoài project, cần tự quản lý xuất artifact.

Job Kaggle mới không tự thấy output của job trước. `.krunignore` loại artifacts/NPZ/trọng số local khỏi gói mã. Khi thử fold 1, có thể chạy 01 rồi 02 trong cùng session. Nếu chạy job riêng, mount/upload artifact cũ rồi đặt `ARTIFACT_INPUT_ROOT` trong cell setup, hoặc biến môi trường `PPGCVAE_INPUT_ROOT`, tới thư mục chứa `prepared`, `proxy`, `tuning`, v.v. Notebook sao chép chúng sang OUTPUT_ROOT có quyền ghi.

KRun mặc định timeout 600 giây mỗi cell. Tăng `--cell-timeout` cho train/tuning, ví dụ:

```powershell
krun run notebooks/01_build_manifest_and_preprocess.ipynb --dataset ameersifat53/ppg-dalia-dataset --cell-timeout 7200
```

Lệnh này chỉ chạy được sau khi khai báo offset/provenance trong notebook 01. Chọn accelerator GPU cho 02–06 khi chạy dữ liệu đầy đủ. `krun.yaml` giữ entrypoint audit trên CPU để kiểm tra dữ liệu trước.

## Sinh trên CPU

```powershell
$env:PYTHONPATH = "$PWD\src"
$resultRoot = '.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'
# Nếu đang dùng gói bàn giao đã giải nén: $resultRoot = 'results'
.venv\Scripts\python.exe -m ppg_cvae.cli.generate --checkpoint "$resultRoot/cvae/fold_01/seed_42/decoder_state_dict.pt" --hr 80 --seed 42 --num-samples 10 --detector-config "$resultRoot/proxy/fold_01/detector_config.json" --output artifacts/demo/generated_ppg.npz
```

CLI lưu waveform, HR đo được và cờ thất bại. Truyền `--condition-mode zero` cho decoder không điều kiện; latent khác 16 cần `--latent-dim`. Sinh tự do không cần encoder/proxy. Đầu ra là biên độ chuẩn hóa, không phải điện áp cảm biến.

## Giới hạn kiểm chứng

Thực nghiệm thật đã hoàn tất trên Kaggle. MAE cVAE tại 60/80/120 bpm lần lượt khoảng 17,88/8,47/34,09 bpm: chưa đạt mục tiêu ≤3 bpm, dù V gần 100%. Gaussian có MAE chung khoảng 3,32 bpm so với 20,15 bpm của cVAE. Sai số proxy và detector trên dữ liệu thật là giới hạn cần đọc cùng kết quả sinh.

Decoder FP32 thực tế 340.199 byte; benchmark CPU Kaggle và benchmark CPU local được báo cáo riêng. `artifacts/completion/local_cpu_environment.json` ghi cấu hình máy, RAM và phạm vi số đo bộ nhớ; `local_cpu_latency.csv` tách decoder khỏi lấy mẫu z + decoder + đo HR. Không suy các số đo này sang ESP32 hoặc INT8. Bộ lọc SOS và sosfiltfilt là xử lý ngoại tuyến.

## Hoàn thiện và bàn giao

`scripts/complete_delivery.py` tạo bảng bổ sung từ artifact đã khóa; `scripts/audit_motion.py` đọc ACC thô để kiểm tra chuyển động, không thay đổi tập hợp lệ hoặc chọn mô hình. KL từng chiều trong artifact bổ sung là snapshot validation tại checkpoint cuối; lịch sử epoch cũ không được dựng lại. Mã train mới ghi KL từng chiều cho các lần chạy sau.

Báo cáo Word ở `reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx`. Gói bàn giao gồm source, notebook, file môi trường, checkpoint, cấu hình, bảng/hình và báo cáo; không gồm dữ liệu cảm biến thô hoặc toàn bộ NPZ sinh lớn. Dataset và lệnh tái tạo được ghi kèm. Không cần slide hoặc vi điều khiển theo phạm vi tài liệu yêu cầu; web demo là tùy chọn. Kết quả chưa đạt mục tiêu HR được giữ nguyên, không huấn luyện lại để thay đổi kết luận sau khi đã xem test.
