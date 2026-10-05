# Căn nhãn PPG DaLiA cho notebook 01

Ngày kiểm tra: 04/10/2026. Dataset đang dùng: `ameersifat53/ppg-dalia-dataset`, các tệp đồng bộ `PPG_FieldStudy/SX/SX.pkl`.

Đặt `LABEL_START_SAMPLE = 0`. Nhãn `label[j]` được ghép với `BVP[128*j : 128*j+512]`: cửa sổ 8 giây ở 64 Hz, bước 2 giây. Mốc 4 giây là tâm cửa sổ đầu tiên, không phải chỉ số bắt đầu cửa sổ.

Nguồn gốc là [readme chính thức của UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf), mục I.2 trang 1 và III.3 trang 3–4. Readme mô tả SX.pkl đã đồng bộ, nhãn HR từ ECG trên cửa sổ 8 giây dịch 2 giây, và yêu cầu chia BVP với cùng cửa sổ. Bản tải về được giữ tại `PPG_DaLiA_readme.pdf`, SHA256 `018a3bab7f1d02261ad4a42720f2235ac20e1c3c0a052f02ef91208096d4fc1b`.

Readme không viết trực tiếp biến s0=0. Giá trị này được suy ra từ cách chia tín hiệu đồng bộ từ đầu bản ghi và đối chiếu audit thực tế, không chọn bằng MAE hoặc kết quả test.

Audit ở `.krun/jobs/20261003-171957-ff75c8/output/krun_outputs/artifacts/audit/dataset_audit.csv` có đủ 15 người duy nhất. Với từng người, số nhãn bằng `floor((N_ppg - 512)/128) + 1`. Giao các khoảng offset không âm cho phép giữ nguyên số cửa sổ của cả 15 người chỉ chứa 0. Điều này hỗ trợ mốc chung bằng 0 khi không cắt bớt đầu/cuối tín hiệu.

Phạm vi: áp dụng cho các SX.pkl đồng bộ đã audit. Nếu thay dataset bằng tín hiệu raw CSV hoặc bản đã cắt/resample, phải xác minh lại; số lượng khớp tự nó không chứng minh mọi nhãn đã đúng thời gian. Notebook vẫn kiểm tra lại số nhãn từng người khi preprocessing và giữ chỉ số gốc khi loại cửa sổ.
