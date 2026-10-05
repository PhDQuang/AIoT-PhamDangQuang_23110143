# Rà soát project PPG cVAE ngày 03 tháng 10 năm 2026

Project chưa hoàn thành các thí nghiệm đã dự kiến. Kiến trúc và chia fold đúng đặc tả, nhưng EDA cũ có lỗi nhận dạng người, nhiều notebook chưa chạy xử lý hoặc huấn luyện thật, và benchmark 08 dùng model ngẫu nhiên. Đã sửa mã và nối các bước thành luồng chạy có artifact. Chưa chạy lại trên dataset thật.

## Bằng chứng từ các bước trước

Đã đối chiếu mã nguồn, cấu hình, notebook, test, đề cương DOCX, tài liệu Markdown và metadata/output/log các job trong `.krun/jobs`. Các bản sao lịch sử trong `.krun` được giữ nguyên.

| Bước cũ | Bằng chứng | Đánh giá |
|---|---|---|
| Audit 00 | Job 105008 complete; chỉ tìm file, chưa kiểm tra mã người duy nhất | Chạy lại audit đã sửa |
| EDA | `111909-2a5ca8/output/krun_outputs/artifacts/figures/dataset_summary.csv` có 15 dòng nhưng 7 dòng S1, thiếu S10–S15 | Bảng/hình theo người và fold phải tính lại |
| Preprocess 01 | Job 112442 complete nhưng cell chỉ load dữ liệu và để comment yêu cầu xử lý | Chưa có manifest/normalization/array dùng train |
| Proxy 02 | Job 113031 complete nhưng chỉ in hướng dẫn gọi hàm | Chưa có checkpoint hoặc độ chính xác proxy thật |
| Tuning 03 | Job 113929 complete nhưng chỉ in năm cấu hình | Chưa chọn beta/lambda bằng thực nghiệm |
| Final/ablation/evaluation | Notebook cũ chủ yếu khung; artifacts local chỉ có `.gitkeep` | Chưa đủ kết quả báo cáo |
| Benchmark 08 | Job 124502 tạo `ConditionalVAE()` mà không load checkpoint | Chỉ đo kiến trúc khởi tạo, chưa xuất decoder đã train |

Lỗi S10–S15 do kiểm tra substring S1 trước S10. Dictionary EDA theo subject cũng bị ghi đè. Kiểm tra giao tập trong `splits.py` đúng nhưng không phát hiện loader đặt sai tên người.

## Những phần đã sửa

- Loader nhận đầy đủ token S1–S15, sắp xếp theo số, dừng khi trùng file/người và đọc MAT nested struct. EDA kiểm tra đủ người duy nhất, đếm cửa sổ riêng từng bản ghi.
- Thêm `data/prepare.py`: dùng PPG[s0 + 128j : s0 + 128j + 512], bắt buộc khai báo offset và nguồn xác nhận, kiểm tra số nhãn. Loại mẫu không đánh lại chỉ số nhãn.
- Lọc từng đoạn hữu hạn riêng, ghi lý do loại, không nối qua NaN/Inf. Margin biên cố định 128 mẫu. Mean/std chỉ dùng điểm train thuộc cửa sổ hợp lệ, không đếm lặp phần chồng lấp.
- Validation proxy/cVAE theo batch, checkpoint có bản sao độc lập trên CPU, seed và lưu lịch sử/config/environment. Loss prior tùy chọn nhận mẫu prior thật. Patience dừng sớm tính sau warm-up.
- A3 tính trên toàn bộ mẫu, gồm thất bại. Hình thái dùng nhịp chân–chân, sửa rise time âm; diversity căn chân xung thay vì đỉnh.
- Detector được chọn và lưu bằng real validation. Notebook proxy xuất sai số theo người và dải HR.
- `workflow.py` thực hiện tuning theo V/MAE/A3/RMSE, train cấu hình đã chọn, fit ngân hàng Gaussian trên nhịp train, ablation và đánh giá cuối. Tuning không đọc test.
- Notebook/CLI thực hiện hành động thật. Bước 07 yêu cầu khóa quy trình, bước 08 nạp checkpoint thật. Tổng hợp seed trong fold trước khi tổng hợp fold.
- Output về `artifacts/` đúng cấu hình KRun; có cell nhập artifact đã mount. Sửa `requirements-kaggle.txt` vốn viết nhiều package trên một dòng không hợp lệ.
- Benchmark dùng bản sao decoder CPU FP32, giữ mode/device model gốc và khôi phục số luồng. Lưu byte tệp, CPU/GPU/RAM và phiên bản thư viện.

## Kiểm chứng

37 test qua trên CPU, gồm nhận diện 15 người, duplicate/schema MAT, căn nhãn, tránh leakage normalization, khoảng thiếu, checkpoint CPU, A3, rise time, phase alignment, fit Gaussian và tổng hợp fold/seed. Test tích hợp dùng dữ liệu mô phỏng, ngân sách một epoch và số mẫu nhỏ để đi qua preprocessing → proxy → detector → năm cấu hình tuning → checkpoint → đánh giá cVAE/Gaussian.

Cú pháp Python và code trong 10 notebook hợp lệ; 10 notebook qua schema nbformat. Mọi code cell EDA đã chạy thành công với dữ liệu mô phỏng 15 người; sửa tên tham số boxplot cho Matplotlib mới và lọc liên tục trước khi cắt các đoạn minh họa. Đã render và xem hình 20 mẫu mô phỏng. KRun dry-run notebook 01 đóng gói được project, không gửi job và không tạo thí nghiệm thật.

Các kiểm tra này xác nhận logic và đường dẫn artifact, chưa xác nhận hiệu năng PPG-DaLiA/CUDA, độ chính xác proxy/detector hay mục tiêu MAE/V/P95.

## Bước tiếp theo

1. Chạy lại audit/EDA. Kiểm tra S1–S15 xuất hiện đúng một lần và số mẫu/nhãn từng người.
2. Xác nhận s0 bằng readme của bản dataset đang mount, rồi khai báo `LABEL_START_SAMPLE` và `ALIGNMENT_SOURCE` ở notebook 01. Không chọn offset bằng cách thử đến khi MAE tốt hơn. [Trang UCI gốc](https://archive.ics.uci.edu/dataset/495/ppg+dalia) chỉ dẫn readme về cấu trúc dữ liệu/nhãn; [Deep PPG](https://kristofvl.github.io/usi/pdf/ubi_sensors2019.pdf) mô tả cửa sổ 8 giây, bước 2 giây.
3. Chạy preprocessing fold 1. Kiểm tra manifest, normalization, ba NPZ, `subject_summary.csv`, `hr_coverage.csv` và lý do loại. Không đi tiếp chỉ vì kernel complete.
4. Chạy proxy/detector fold 1. Xem MAE theo người/dải HR. Chốt tiêu chí chấp nhận proxy trước khi mở test; nếu proxy sai lớn, giữ đối chứng lambda=0 và báo giới hạn.
5. Kiểm tra một run nhỏ trước full tuning. Giữ batch/budget nhất quán giữa cấu hình. Job Kaggle riêng cần mount artifact trước; `.krunignore` không đưa dữ liệu/checkpoint local vào gói mã. Tăng `--cell-timeout` cho cell train dài.
6. Khi fold 1 ổn, chạy preprocessing/proxy/tuning mọi fold, train 5×3 seed, Gaussian và ablation. Khóa mọi quy tắc trước khi mở test, sau đó chạy 07, 08 và tổng hợp.

Bảng/hình báo cáo cuối cần được xem sau chạy thật. Bộ hình hiện có 20 mẫu và các lỗi; cần hoàn thiện biểu đồ HR theo điều kiện, giữ z đổi HR, phân bố hình thái và độ phủ khi có checkpoint thật. Proxy chưa có ngưỡng chấp nhận thực nghiệm; offset chưa được xác minh từ readme đang mount. Chưa có căn cứ kết luận đạt chỉ tiêu hay chạy được trên thiết bị IoT đích.
