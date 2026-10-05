# Kết quả bước 03 và bước chạy tiếp

Job `20261004-065805-ebc5a4` hoàn tất; toàn bộ output đã tải thành công bằng `scripts/download_krun_output.py`. Đã mở kiểm tra 12 checkpoint.

Kiểm tra tiếp phát hiện bản sao `krun_outputs/artifacts/prepared/fold_01/train.npz` rỗng trong output tải lại. Đã khôi phục từ artifact gốc bước 01 (job `20261003-182853-1b7400`), ghi đường dẫn và SHA256 trong `local_output_recovery.json` của job tuning. Notebook 04 lấy prepared từ dataset Kaggle; đã kiểm tra bootstrap và đọc train/val thật từ artifact gốc, proxy, checksum input và cấu hình chọn. Chưa chạy huấn luyện bước 04.

Grid chọn `beta=0.01`, `lambda_hr=0.1` bằng quy tắc validation đã định trước. Không thay đổi lựa chọn bằng test.

| Validation, trung bình 3 HR | Cấu hình chọn | Đối chứng lambda_hr=0 |
|---|---:|---:|
| MAE HR (bpm) | 19,69 | 21,74 |
| A3 | 7,83% | 12,17% |
| RMSE reconstruction | 0,555 | 0,545 |

Ở cấu hình chọn, MAE tại HR 60/80/120 lần lượt 22,46 / 7,36 / 29,26 bpm; A3 lần lượt 0% / 22% / 1,5%. Loss HR cải thiện MAE trung bình nhưng chưa cải thiện A3 và làm RMSE tăng so với đối chứng. V=100% chỉ nói detector trả được HR, không chứng minh HR đúng. Chưa đủ bằng chứng kết luận mô hình đạt chất lượng mong muốn, hoặc loss HR có lợi trên mọi chỉ số. Proxy/detector trên real validation cũng có sai số đáng kể; chưa thể quy toàn bộ lỗi cho bộ sinh.

Notebook 04 hiện chạy fold 1 với seed 42/43/44 để kiểm tra độ ổn định của cấu hình chọn. Đây là bước tiếp theo của thử nghiệm, chưa phải hoàn tất cross validation 5 fold hay công bố chất lượng cuối cùng. Sau đó cần xem summary từng seed và waveform/hình thái trên validation, cùng baseline/ablation; test vẫn đóng.

Gói nhỏ `inputs/final/fold_01` gồm proxy, detector, metadata chất lượng và kết quả lựa chọn tuning. Có SHA256 và ID job nguồn; notebook kiểm tra gói trước khi huấn luyện. Dữ liệu prepared vẫn lấy từ dataset đã tạo.

```powershell
krun run notebooks/04_train_final_cross_validation.ipynb --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-fold01 --input inputs/final/fold_01 --cell-timeout 21600 --timeout 43200 --detach
```

Ghi lại job ID mà lệnh trả về. Theo dõi bằng `krun status <JOB_ID>`. Khi trạng thái COMPLETE, tải với thời gian truyền dài để tránh giới hạn 120 giây của KRun:

```powershell
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py <JOB_ID> --transfer-timeout 1800
```

Kết quả chính ở `.krun/jobs/<JOB_ID>/output/krun_outputs/artifacts/cvae/fold_01/`: mỗi seed có checkpoint, training_history.csv, validation_metrics.json, prior_validation_summary.csv, resolved_config.yaml; training_plan.json ghi cấu hình và nguồn tuning.

Fold 2–5 chưa có prepared/proxy/tuning trong dataset fold 1. Cần tạo input và chọn cấu hình riêng cho từng fold trước khi chạy chúng; không dùng lại normalization, proxy hoặc cấu hình chọn của fold 1 như thể đã thực hiện validation riêng.
