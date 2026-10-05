# Chạy bước 05 cho fold 1

Khi chuẩn bị bước này, job cVAE `20261004-085030-b15249` vẫn RUNNING. Baseline chỉ cần prepared train và detector nên có thể chạy trong lúc chờ, bằng CPU.

Notebook 05 fit một bank Two-Gaussian từ tối đa 500 cửa sổ train, fit seed 42. Ba generation seed 42/43/44 dùng cùng bank, mỗi seed sinh 200 mẫu tại mỗi HR 60/80/120 để lưu diagnostic. Không dùng checkpoint cVAE và không đọc test. Baseline tuân theo bộ detector đã chọn từ real validation; V cao chưa đảm bảo HR chính xác.

```powershell
krun run notebooks/05_train_evaluate_gaussian_baseline.ipynb --gpu cpu --dataset hcsinhgoethe/ppg-cvae-prepared-fold01 --input inputs/final/fold_01 --cell-timeout 7200 --timeout 10800 --detach
```

Ghi job ID và kiểm tra `krun status <JOB_ID>`. Khi COMPLETE:

```powershell
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py <JOB_ID> --transfer-timeout 1800
```

Output chính: `artifacts/gaussian/fold_01/baseline_plan.json`, và `seed_42`, `seed_43`, `seed_44`, mỗi thư mục có `gaussian_fit.json`, `prior_validation_summary.csv`. Bảng diagnostic này chưa phải kết quả test cuối cùng.

Trước khi sang ablation bước 06, kiểm tra cVAE bước 04 đã hoàn tất và tải kết quả. Chưa mở bước 07: fold 2–5 và protocol cuối cùng vẫn chưa hoàn tất.
