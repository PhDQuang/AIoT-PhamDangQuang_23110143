# Chạy bước 06: ablation fold 1

Đã xác nhận hai job `20261004-085030-b15249` (cVAE) và `20261004-090009-492f73` (Gaussian) COMPLETE trên Kaggle. Bước 06 sử dụng dữ liệu prepared, proxy/detector và cấu hình chọn đã có trong `inputs/final/fold_01`; không cần mang checkpoint cVAE sang job huấn luyện ablation.

Đã đọc summary 200 mẫu/HR tại 60/80/120 cho 3 seed và mở thành công 6 checkpoint cVAE. Diagnostic hiện tại:

| Seed | cVAE MAE bpm | cVAE A3 | Gaussian MAE bpm | Gaussian A3 |
|---|---:|---:|---:|---:|
| 42 | 19,69 | 7,83% | 3,22 | 94,17% |
| 43 | 19,39 | 8,50% | 3,01 | 94,33% |
| 44 | 20,59 | 10,17% | 3,49 | 93,50% |

Đây là trung bình các target trong diagnostic sinh, chưa phải đánh giá test. Gaussian hiện tốt hơn theo bộ detector này; chưa kết luận về hình thái hay chất lượng tổng thể. Cần dùng ablation để kiểm tra tác động của loss HR/conditioning trước khi mở rộng hoặc chốt kết quả. Bảng số đã lưu tại `artifacts/review/fold_01_before_ablations.csv`.

Giữ beta=0.01, seed=42, kiến trúc, optimizer, batch size, warmup và budget như bước 04:

| Run | Condition | lambda_hr |
|---|---|---:|
| cVAE tham chiếu | HR | 0.1 |
| no_hr_loss | HR | 0 |
| no_condition | zero | 0 |

So sánh cVAE với no_hr_loss để xét loss HR; so sánh no_hr_loss với no_condition để xét conditioning. So sánh trực tiếp cVAE với no_condition thay đổi cả hai yếu tố. Không đọc test.

```powershell
krun run notebooks/06_run_ablations.ipynb --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-fold01 --input inputs/final/fold_01 --cell-timeout 21600 --timeout 43200 --detach
```

Kiểm tra `krun status <JOB_ID>`. Khi COMPLETE:

```powershell
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py <JOB_ID> --transfer-timeout 1800
```

Kết quả chính: `artifacts/ablations/no_hr_loss/fold_01/seed_42` và `artifacts/ablations/no_condition/fold_01/seed_42` có checkpoint, config, history và prior_validation_summary.csv. `artifacts/ablations/fold_01` có ablation_plan.json và prior_validation_comparison.csv.

Đã kiểm tra notebook schema, syntax và chạy cả hai nhánh trên train thật với budget giảm (32 train, 16 val, 1 epoch, 4 mẫu/HR) để kiểm tra nối input, config và checkpoint. Đây không phải kết quả chất lượng của run đầy đủ.

Sau bước 06 cần đánh giá validation/hình thái và hoàn thiện fold 2–5. Chưa mở notebook 07 để đánh giá test cuối cùng khi mới có fold 1.
