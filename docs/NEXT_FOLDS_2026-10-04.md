# Sau ablation fold 1

Job `20261004-090952-5512e0` COMPLETE, download COMPLETE. Đã đọc checkpoint và config của hai ablation: beta=0.01, seed=42, lambda_hr=0; condition_mode lần lượt hr và zero.

Diagnostic 200 mẫu tại mỗi HR 60/80/120, seed 42:

| Phương pháp | MAE HR trung bình bpm | A3 trung bình |
|---|---:|---:|
| cVAE, lambda_hr=0.1 | 19,69 | 7,83% |
| no_hr_loss | 21,74 | 12,17% |
| no_condition, không loss HR | 21,93 | 9,50% |
| Two-Gaussian | 3,22 | 94,17% |

Loss HR giảm MAE trung bình so với no_hr_loss nhưng giảm A3. Conditioning không loss HR chỉ thay đổi ít MAE so với unconditional. Đây là diagnostic validation/generation, không phải test và không chứng minh toàn bộ chất lượng hình thái. Vẫn giữ cấu hình đã chọn theo quy tắc đề cương; không thay đổi bằng test.

Để hoàn tất quy trình 5 fold, bước tiếp theo là chuẩn bị fold 2–5 từ dữ liệu **raw**. Dataset `hcsinhgoethe/ppg-cvae-prepared-fold01` chỉ có fold 1 nên không đủ cho bước này.

Notebook mới `01b_prepare_remaining_folds.ipynb` giữ alignment/preprocessing đã xác minh, chạy CPU và fit normalization riêng từ train của từng fold. Không huấn luyện hoặc đánh giá mô hình trên test. Lưu split arrays và ZIP để dùng làm dataset prepared mới.

```powershell
krun run notebooks/01b_prepare_remaining_folds.ipynb --gpu cpu --dataset ameersifat53/ppg-dalia-dataset --cell-timeout 14400 --timeout 21600 --detach
```

Ngay khi có job ID, thay JOB_ID bằng giá trị thật và chạy lệnh dưới. Lệnh chờ job hoàn tất rồi **tự tải** output, không cần kiểm tra status thủ công:

```powershell
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py JOB_ID --wait --wait-timeout 21600 --transfer-timeout 3600
```

Giữ terminal chạy lệnh chờ/tải. Nếu đóng terminal, job vẫn chạy trên Kaggle; có thể chạy lại lệnh với cùng ID, không gửi lại job. Không cần `--wait` khi chỉ tải job đã COMPLETE.

ZIP sau khi tải: `.krun/jobs/JOB_ID/output/krun_outputs/artifacts/ppg-cvae-prepared-folds02-05.zip`. Khi hoàn tất, đưa ZIP lên dataset Kaggle prepared mới, sau đó chuẩn bị proxy/detector, tuning, final seeds, Gaussian và ablations cho từng fold 2–5. Chưa chạy notebook 07 cho đến khi đủ fold và chốt protocol.

Notebook mới đã được kiểm tra syntax/schema và thực thi đầy đủ trên bộ dữ liệu tổng hợp 15 subject để xác nhận tách 4 fold và tạo ZIP. Kiểm tra này không thay thế việc chuẩn bị raw PPG-DaLiA trên Kaggle.

## Sau khi job chuẩn bị hoàn tất

Job chuẩn bị `20261004-125540-b4ff10` đã COMPLETE và download COMPLETE. Đã kiểm tra CRC của ZIP và các file chính của fold 2–5. Thư mục upload sẵn tại `artifacts/kaggle_upload/folds02-05`, gồm ZIP và dataset-metadata.json. Lệnh upload tạo dataset private theo mặc định:

```powershell
$kaggleExe = Join-Path $env:APPDATA 'uv/tools/krun-cli/Scripts/kaggle.exe'
$uploadDir = (Resolve-Path -LiteralPath 'artifacts/kaggle_upload/folds02-05').Path
& $kaggleExe datasets create -p $uploadDir
& $kaggleExe datasets status hcsinhgoethe/ppg-cvae-prepared-folds02-05
```

Chờ status `ready`, rồi chạy notebook mới 02b để train proxy và calibrate detector riêng cho mỗi fold:

```powershell
krun run notebooks/02b_train_remaining_proxies.ipynb --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-folds02-05 --cell-timeout 21600 --timeout 43200 --detach
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py JOB_ID --wait --wait-timeout 43200 --transfer-timeout 3600
```

Đã kiểm tra notebook 02b bằng dữ liệu tổng hợp 15 subject, training 1 epoch: cả 4 fold tạo được proxy checkpoint, detector config và bảng real validation. Cần xem chất lượng proxy thật sau job 02b trước tuning từng fold.

Trên Windows, dùng đường dẫn tuyệt đối từ Resolve-Path khi upload. Bản Kaggle CLI đang cài chỉ thay dấu phân cách Windows `\` khi tạo tên file trạng thái resumable upload; truyền thư mục tương đối có `/` gây lỗi `[Errno 2]` tại `.kaggle/uploads/...json`. ZIP prepared không bị lỗi trong trường hợp này.

Đã upload thành công bằng đường dẫn tuyệt đối. Dataset private `hcsinhgoethe/ppg-cvae-prepared-folds02-05` đã có trạng thái `ready`; không cần chạy lại `datasets create`.
