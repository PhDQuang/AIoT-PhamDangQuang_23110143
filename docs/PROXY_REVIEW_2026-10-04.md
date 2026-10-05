# Proxy fold 1 và chuẩn bị tuning cVAE

Job `20261004-050604-c55386` đã hoàn tất notebook 02 trên GPU T4. Đã tải checkpoint, detector và bảng validation. Các số sau là kết quả thật của fold 1, không phải test hoặc kết quả sinh cVAE.

| Đại lượng | Kết quả |
|---|---:|
| Proxy validation MAE | 11,89 bpm |
| Proxy MAE S4 | 6,96 bpm |
| Proxy MAE S5 | 16,54 bpm |
| Proxy MAE S6 | 12,22 bpm |
| Proxy MAE HR [120,140) | 15,24 bpm |
| Proxy MAE HR [140,160) | 31,04 bpm |
| Detector MAE trên real validation | 9,10 bpm |
| Detector V | 99,99% |
| Detector A3, tính trên mọi cửa sổ | 43,36% |

Detector đã chọn prominence=0.1, min_distance=20 bằng quy tắc validation hiện có. V cao không chứng minh detector đo chính xác: A3 và MAE cho thấy sai số đáng kể. Proxy cũng chưa đủ chính xác để coi loss HR là phép kiểm chứng độc lập, đặc biệt tại HR cao.

Notebook 03 giữ nguyên năm cặp beta/lambda của đề cương. Trước grid đó, chạy thêm đối chứng có điều kiện beta=0.01, lambda_hr=0, seed 42, cùng budget tối đa 100 epoch và cùng quy tắc validation. Đối chứng này được ghi riêng và không tham gia lựa chọn giữa năm ứng viên. So sánh nó với các ứng viên beta=0.01 để xem tác động của loss HR. Đây là diagnostic trước test, không thay thế hai ablation bắt buộc sau khi chốt beta.

Các số HR trên mẫu sinh ở bước 03 cần được đọc kèm hình thái/đa dạng và giới hạn bộ đo. Không diễn giải MAE thấp theo proxy là HR đúng. Không mở test hoặc đổi detector dựa trên kết quả test.

Để chuyển artifact sang job tuning, checkpoint và các bảng nhỏ được sao chép nguyên trạng vào `inputs/proxy/fold_01/`, kèm SHA256 và ID job nguồn trong `source_manifest.json`. Thư mục này bị ignore mặc định và chỉ đưa lên KRun khi truyền `--input`. Dữ liệu prepared lớn tiếp tục lấy từ dataset đã tạo, không phải upload lại.

```powershell
krun run notebooks/03_tune_cvae.ipynb --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-fold01 --input inputs/proxy/fold_01 --cell-timeout 21600 --timeout 43200
```

Kết quả cần xem trong `artifacts/tuning/fold_01/`: `candidate_comparison.csv`, `selected_config.json`, `comparison_with_no_hr_control.csv`, `measurement_quality.json` và history/prior validation của từng run. Chưa chuyển sang final training chỉ vì job complete; cần đánh giá các bảng và đối chứng trước.
