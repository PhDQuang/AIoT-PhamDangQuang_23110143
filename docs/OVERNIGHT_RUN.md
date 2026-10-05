# Chạy toàn bộ phần còn lại trên Kaggle khi tắt máy

Job đã gửi: `20261004-151954-483f31`; đã xác nhận RUNNING trên T4. URL: https://www.kaggle.com/code/hcsinhgoethe/final-iot-for-ai-20261004-151954-483f31

Script cũ `scripts/run_remaining_tuning.ps1` điều phối từ máy Windows nên không gửi được các job tiếp theo nếu tắt máy. Script mới `scripts/run_overnight.py` chạy toàn bộ phần còn lại trong một job GPU Kaggle; sau khi submit thành công, máy cá nhân không cần duy trì kết nối.

Đã dừng script điều phối cục bộ để tránh gửi thêm job riêng. Job tuning fold 2 `20261004-145556-c71259` đã COMPLETE; đã tải các bảng lựa chọn cần thiết và mang chúng vào gói chạy mới. Chưa tải toàn bộ output của job riêng này nên download_status của nó có thể vẫn pending.

Gói `inputs/overnight` có proxy/detector 5 fold, tuning fold 1/2 và checkpoint cVAE 3 seed, Gaussian, ablation của fold 1. Bundle manifest lưu nguồn job và SHA256. Script dùng prepared từ 2 dataset đã tạo, không đưa prepared lớn vào source.

Thứ tự thực thi trên Kaggle:

1. Tuning fold 3–5 với grid cũ và đối chứng no-HR riêng.
2. Huấn luyện cVAE seed 42/43/44, fit Gaussian và chạy 2 ablation cho fold 2–5.
3. Lưu snapshot protocol, cấu hình chọn, detector, normalization và checkpoint SHA256 cho đủ 5 fold trước khi đọc test. Không tuning sau test.
4. Đánh giá 1000 mẫu/HR tại 60/80/120, đủ 40 method/fold/seed run. Tổng hợp seed trước, fold sau; lưu 120 dòng generation summary.
5. Xuất decoder và CPU benchmark FP32 single-thread với 100 warmup/1000 iteration cho 15 checkpoint cVAE. Đây là CPU máy Kaggle, không phải laptop hay vi điều khiển.

Output chính:

- `artifacts/overnight/progress.json`: stage hiện tại, các stage đã xong, status cuối.
- `artifacts/overnight/locked_evaluation_protocol.json`: protocol trước test.
- `artifacts/summary/cross_fold_summary.csv`, `seed_summary_by_fold.csv`, `all_runs.csv`.
- `artifacts/summary/resource_metrics.csv`.
- `artifacts/figures/cross_fold_generation.png` và hình waveform/failure riêng của mỗi run.
- Các checkpoint/history/config trong `cvae`, `gaussian`, `ablations`, `tuning`.

Lệnh gửi một job (không chạy lại nếu job đã được gửi):

```powershell
krun run scripts/run_overnight.py --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-fold01 --dataset hcsinhgoethe/ppg-cvae-prepared-folds02-05 --input inputs/overnight --timeout 43200 --detach
```

Sau khi job được Kaggle chấp nhận, có thể tắt máy. Kết quả được lưu trên Kaggle; tải về khi mở máy:

```powershell
krun status JOB_ID
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py JOB_ID --wait --wait-timeout 43200 --transfer-timeout 7200
```

Với job đã gửi, thay JOB_ID bằng `20261004-151954-483f31`. Không chạy lại lệnh submit chỉ để xem trạng thái hoặc tải output.

Nếu job ERROR, tải output bằng cùng script bỏ `--wait` để xem progress/traceback và checkpoint đã lưu. Không gửi lại toàn bộ job trước khi kiểm tra nguyên nhân và artifact có thể tái sử dụng.

Đã kiểm tra bundle hashes, 10 checkpoint input, syntax và 38 test. Test mới chạy chuỗi tuning → train → baseline/ablation → khóa protocol → test → aggregate/figures → benchmark với 2 fold, budget giảm; chưa thay thế run thật. Thời gian hoàn tất phụ thuộc GPU quota, hàng chờ, số epoch và lỗi dịch vụ. Script không viết báo cáo hoặc slide.
