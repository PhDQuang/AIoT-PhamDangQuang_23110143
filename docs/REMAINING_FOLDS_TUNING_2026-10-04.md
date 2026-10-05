# Tuning fold 2–5

Job proxy `20261004-142710-0769e3` đã COMPLETE và download COMPLETE. Đã mở 4 checkpoint và kiểm tra input thật cho cả 4 notebook tuning.

| Fold | Proxy val MAE bpm | Detector val MAE bpm | Detector A3 |
|---|---:|---:|---:|
| 2 | 7,38 | 11,65 | 31,22% |
| 3 | 8,56 | 10,03 | 42,64% |
| 4 | 5,83 | 7,37 | 46,89% |
| 5 | 6,59 | 9,75 | 38,56% |

Proxy/detector có sai số đáng kể. Giữ grid/rule như fold 1, xem HR scores là exploratory và đối chiếu no-HR control, hình thái/đa dạng. Không sử dụng test để chọn cấu hình.

Mỗi notebook `03_tune_cvae_fold_02` đến `03_tune_cvae_fold_05` chạy 5 candidate và một control beta=.01, lambda_hr=0 riêng. Notebook lấy prepared từ dataset `hcsinhgoethe/ppg-cvae-prepared-folds02-05`, proxy đúng fold từ `inputs/proxy/fold_XX`, kiểm tra SHA256 và nguồn job.

Chạy tất cả tuần tự, chờ và tự tải sau mỗi fold:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_remaining_tuning.ps1
```

Chỉ chạy fold 2 nếu muốn kiểm tra trước:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_remaining_tuning.ps1 -Folds 2
```

Giữ terminal mở. Mỗi job dùng T4, cell timeout 21600, wait timeout 43200, transfer timeout 3600. Script dừng nếu submit/chờ/tải lỗi và không tự gửi lại job lỗi. Bypass chỉ áp dụng cho tiến trình PowerShell này.

Nếu bị ngắt sau khi submit, dùng ID đã in ra để chờ/tải lại với `scripts/download_krun_output.py JOB_ID --wait`, rồi chỉ chạy các fold chưa được gửi. Không chạy lại toàn bộ script như cơ chế resume: mặc định nó sẽ gửi job mới cho mỗi fold được yêu cầu.

Sau khi hoàn tất, cần xem candidate_comparison.csv, selected_config.json, comparison_with_no_hr_control.csv và measurement_quality.json của từng job trước khi chuẩn bị final training 3 seed cho fold 2–5.

Đã kiểm tra notebook syntax/schema, input arrays thực, proxy weights, detector và checksum cho cả 4 fold; đã kiểm tra cú pháp PowerShell. Chưa gửi job tuning trong lượt chuẩn bị này.
