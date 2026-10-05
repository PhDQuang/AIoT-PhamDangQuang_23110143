# Đối chiếu yêu cầu và kết quả project — 05/10/2026

## Cập nhật sau khi hoàn thiện hồ sơ

Các nhận xét trong ma trận bên dưới là trạng thái lúc kiểm tra ban đầu. Sau yêu cầu hoàn thiện, đã bổ sung:

- `artifacts/completion/`: bảng trước/sau theo split, HR min/max, bin 20 bpm, độ phủ điều kiện và bảng 120 run gắn độ phủ; median/failures và kiểm chuẩn bộ đo tổng hợp theo người.
- Word được chỉnh về thân bài 11 pt, năm học 2026–2027; vẫn 25 trang, 18 hình, 9 phương trình OMML. Đã thêm ví dụ lỗi, kiểm tra cùng z/đổi h và cùng h/đổi z, biến thiên seed, rise/cycle và tỷ lệ trích nhịp; render và xem toàn bộ trang.
- Benchmark local CPU Intel Core i5-1145G7, RAM 31,71 GiB: P95 decoder 0,831–1,480 ms; lấy z + decoder + đo HR 1,320–1,906 ms. Đã lưu OS/runtime và bộ nhớ toàn tiến trình, không gọi đó là RAM riêng decoder.
- Đã chạy CLI decoder trên máy thử và thử lại CLI từ ZIP giải nén ngoài project. README đã cập nhật kết quả thật và lệnh đúng; gói bàn giao có source, môi trường, trọng số, bảng/hình và báo cáo, kèm SHA256.
- KL từng chiều: đã tính snapshot toàn bộ validation tại 15 checkpoint cuối và bổ sung log trong mã train cho lần chạy mới. Không thể khôi phục lịch sử KL từng chiều các epoch cũ.
- Thống kê chuyển động được bổ sung qua job Kaggle riêng `20261005-045644-ae0c99`; trạng thái cuối và bảng kết quả được ghi trong hồ sơ hoàn thiện sau khi tải xong.

Mục tiêu MAE HR ≤3 bpm vẫn chưa đạt. Không thay checkpoint, detector hoặc protocol đã khóa; các bổ sung là phân tích và hoàn thiện bàn giao, không phải thí nghiệm cải tiến cVAE mới.

## Nguồn và phạm vi kiểm tra

- `docs/Quy_dinh_De_cuong_AI_cho_IoT_HCMUTE_Fixed.docx`: quy định chung, mục I–II và đặc tả G2.
- `docs/PhamDangQuang_DeCuong_G2_DaChinhSua_KhoaHoc.docx`: đề cương đã chỉnh sửa, mục 1–4.5 và phụ lục.
- Mã nguồn `src/ppg_cvae`, notebook, script chạy, README; báo cáo `reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx` và nguồn nội dung của báo cáo.
- Kết quả cuối tại `.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts/` (gọi là ROOT bên dưới).

Kết luận: đã thực hiện các thí nghiệm chính, nhưng chưa thể đánh dấu đáp ứng toàn bộ đề cương. Mục tiêu chính xác HR của cVAE chưa đạt; một số thống kê, trình bày và thông tin môi trường còn thiếu. Đây là đánh giá hồ sơ và bằng chứng hiện có, không phải bảo đảm điểm số hay chứng nhận mọi thao tác thủ công trước đây không dùng test.

## Ma trận đối chiếu

| Yêu cầu | Bằng chứng hiện có | Đánh giá |
|---|---|---|
| G2: 1D-cVAE điều kiện HR và Gaussian baseline | `models/cvae.py`, `baselines/gaussian.py`, checkpoint và bank Gaussian | Đã thực hiện |
| 15 người; PPG 64 Hz, cửa sổ 512, stride 128; nhãn ECG và offset rõ | `data/prepare.py`, `docs/data_sources/PPG_DaLiA_alignment.md`, `ROOT/prepared/*/preprocessing_config.json` | Đã triển khai và chạy; không tự dịch nhãn để cải thiện điểm |
| Chia theo người, chuẩn hóa chỉ train, không đếm lặp điểm trong các cửa sổ chồng lấp | `data/splits.py`, `data/prepare.py`, manifest và normalization từng fold | Đã triển khai |
| Thống kê người/cửa sổ trước–sau và HR min/max | `prepared/*/subject_summary.csv`, manifest | Có artifact; báo cáo cần trình bày đủ theo split/fold |
| Phân bố toàn miền theo bin HR 20 bpm, tỷ lệ loại theo lý do và mức chuyển động | Có HR coverage quanh 60/80/120; loader/preparer chưa xuất thống kê ACC/motion cho cửa sổ | Chưa đủ. Coverage quanh ba target không thay thế histogram toàn miền và motion audit |
| Kiểm chuẩn proxy/detector trên PPG thật | `proxy/*/validation_by_subject_id.csv`, `validation_by_hr_bin_lower.csv`, detector config | Đã thực hiện; sai số còn đáng kể. Báo cáo cần giữ đơn vị người khi tổng hợp dữ liệu thật |
| Proxy đóng băng, HR loss khả vi, KL chuẩn hóa; sinh prior riêng tái tạo posterior | Mã training/loss/workflow, RMSE có cặp | Đã triển khai; chưa thấy bảng theo dõi KL từng chiều như mục 4.1 yêu cầu |
| Năm cặp tuning trên val, monitoring prior, warmup và early stopping | `workflow.py`, history, tuning từng fold | Đã thực hiện. Không so loss tổng khác beta/lambda để chọn cấu hình |
| Base và Gaussian: 5 fold × 3 seed × 3 HR × 1.000 cửa sổ | `summary/all_runs.csv`: 120 dòng cho 40 run, gồm cả ablation; 120.000 cửa sổ sinh | Đủ số lượt/mẫu |
| Hai ablation cần làm: no_hr_loss và no_condition, 5 fold seed 42 | `ablations/*`, báo cáo mục 5.8 | Đã thực hiện. Cần diễn giải định lượng trực tiếp no_condition so với no_hr_loss để tách tác động điều kiện |
| MAE, median error, V, A3, failed count từng method/fold/seed/HR | `summary/all_runs.csv`, `generation_samples.csv` | Artifact có đủ chỉ số; Word chưa có bảng chi tiết đầy đủ và đánh dấu support/thin/extrapolation từng fold |
| Trung bình seed trước, fold sau; biến thiên seed trong fold | `evaluation/aggregate.py`, `summary/seed_summary_by_fold.csv` | Artifact đúng; Word thiếu bảng/đồ thị biến thiên seed theo fold |
| Hình thái: rise, width50, rise/cycle, IQR, tỷ lệ trích nhịp; biên độ riêng, diversity căn nhịp | `generated_morphology.csv`, `morphology_comparison.csv`, `test_reference_coverage.csv`; hình morphology/diversity trong báo cáo | Đã tính nhiều thành phần; Word mới minh họa hình thái một fold, chưa trình bày đủ rise/cycle và tỷ lệ trích nhịp |
| 20 cửa sổ chọn trước mỗi HR và ví dụ thất bại | `generated_*_fixed_samples.png` dùng chỉ số 0..19; file failures tạo khi có cửa sổ không đo được | Có artifact. Word cần hình thất bại và đường dẫn bộ hình 20 mẫu. Sai HR dù đo được cũng cần minh họa rõ |
| Giữ z đổi h và giữ h đổi z | `generate_fixed` dùng cùng seed/bộ z cho các HR; dữ liệu và bộ hình nhiều chỉ số có sẵn | Có cơ sở tái dựng phép thử, nhưng Word chưa trình bày hai kiểm tra rõ ràng thành bằng chứng về đáp ứng điều kiện/đa dạng |
| Tham số ≤500k, decoder FP32 ≤2 MB; CPU 1 thread/batch1/warmup100/1000 đo | cVAE 366.625; decoder 84.081; tệp 340.199 byte; 15 resource metrics | Đáp ứng kích thước; đã đo decoder. Trung bình P95 của 15 checkpoint khoảng 0,382 ms trên CPU Kaggle |
| CPU model, RAM, OS/runtime; độ trễ toàn quy trình riêng | OS/runtime có; CPU chỉ `x86_64`; chưa có RAM và latency lấy z + decoder + detector | Chưa đủ mục 4.2.2; kích thước tệp không phải RAM vận hành |
| Chương trình CPU sinh PPG 8 giây từ HR/seed, hiển thị target/measured/failure | `cli/generate.py`: CPU decoder, lưu NPZ và in target, measured, measurable | Có CLI. Chưa thấy ứng dụng web trong project; web không phải yêu cầu bắt buộc |
| Báo cáo 15–25 trang, hình, công thức Word/LaTeX, tài liệu khoa học | Word hiện có 25 trang, 18 hình, 9 OMML và nguồn LaTeX, 4 bài báo cùng các nguồn khác | Có báo cáo; cần bổ sung nội dung thiếu bằng thay thế/rút gọn trong giới hạn trang |
| Font 11 hoặc 12, đơn sắc, TNR, lề và giãn dòng | Báo cáo đơn sắc, TNR, lề/giãn dòng đã đặt; body trong script là 11,5 pt | Cỡ chữ thân bài cần đổi về 11 hoặc 12 và render lại toàn bộ để kiểm tra số trang/layout |
| Nộp toàn bộ source, README, môi trường và trọng số | Source/requirements/pyproject có; trọng số trong thư mục ẩn `.krun` | Cần đóng gói thư mục bàn giao dễ tìm. README đang lỗi thời và lệnh CPU dùng đường dẫn artifact chưa trỏ tới output cuối |

## Mục tiêu hiệu năng chưa đạt

| HR mục tiêu | MAE cVAE, bpm (trung bình seed rồi fold) | Mục tiêu |
|---|---:|---|
| 60 | 17,88 | ≤3 |
| 80 | 8,47 | ≤3 |
| 120 | 34,09 | ≤3 |

V gần 100% chỉ nghĩa detector đo được HR, không nghĩa HR đúng. A3 chung của cVAE khoảng 9,54%. Gaussian có MAE chung khoảng 3,32 bpm, cVAE khoảng 20,15 bpm. Đề cương không bắt buộc cVAE thắng Gaussian, nhưng việc chưa đạt chỉ tiêu HR phải ghi rõ. Có thể bàn giao nghiên cứu với kết quả âm trung thực; không được đổi kết quả này thành tuyên bố đã đạt mục tiêu.

Proxy val MAE khoảng 5,83–11,89 bpm và detector khoảng 7,37–11,65 bpm giữa các fold là giới hạn quan trọng. Đề cương mục 4.5 ưu tiên lambda=0 khi proxy chưa đủ tốt: project có đối chứng lambda=0, nhưng không nên mô tả HR loss là thành phần đã được xác nhận đáng tin cậy. Không có ngưỡng chất lượng proxy được chốt để tự động kết luận đạt/không đạt.

## Những phần bổ sung và giới hạn của từ “cải tiến”

Các đóng góp bổ sung rõ nhất là kỹ thuật vận hành: tự động hóa Kaggle/KRun qua script chạy xuyên đêm, chuyển artifact giữa giai đoạn, khôi phục output khi timeout, snapshot/hash protocol và kiểm tra tính toàn vẹn kết quả tải. Chúng cải thiện khả năng chạy lại, truy vết và bàn giao.

Loader nhiều schema, kiểm tra offset/nhãn, không nội suy qua khoảng thiếu, chuẩn hóa train và decoder CLI là những hiện thực kỹ thuật hữu ích của yêu cầu. Không gọi đây là chứng cứ mô hình chính xác hơn khi chưa có đối chứng.

Năm fold, ba seed, A3, Gaussian, hai ablation, morphology, diversity căn nhịp, prior validation và decoder-only đã có trong đề cương sửa đổi; chúng là phần hoàn thành kế hoạch, không phải đổi mới vượt đề cương. Giảm grid xuống năm cặp cũng đã nằm trong chính tài liệu này.

HR-prior, beta=0, latent 8/32 và INT8 chưa có kết quả đối chứng; ba biến thể đầu được ghi là tùy chọn. Không làm chúng không gây thiếu các ablation bắt buộc. Chưa có bằng chứng cải tiến thuật toán làm tăng độ chính xác HR hay chứng minh downstream utility.

## Thiết bị biên, web và yêu cầu nộp

Đề cương mục 1.2: “Phạm vi triển khai là bộ giải mã FP32 trên CPU máy thử hoặc cổng IoT có cấu hình được ghi rõ; chưa đặt yêu cầu chạy trên vi điều khiển.”

Vì vậy không bắt buộc mua thiết bị hoặc triển khai ESP32. CLI CPU trên máy thử phù hợp phạm vi; nên bổ sung tên CPU, RAM và benchmark toàn quy trình trên máy đó. Kích thước decoder nhỏ và benchmark host cho thấy hướng triển khai trên gateway CPU đáng khảo sát, nhưng chưa chứng minh vận hành trên phần cứng biên cụ thể. RAM, runtime, toán tử ConvTranspose, hệ điều hành và latency cần được đo trên thiết bị đích; checkpoint 0,340 MB không bảo đảm toàn chương trình vừa RAM MCU.

Web demo có thể làm giao diện cho decoder thật chạy CPU: nhập HR/seed, vẽ PPG, hiển thị HR đo độc lập và cờ thất bại. Không thay HR đo bằng HR yêu cầu. Giao diện web không tự tạo bằng chứng triển khai biên; máy chạy backend mới là nơi suy luận.

Quy định mục I và II yêu cầu báo cáo kỹ thuật toàn văn, toàn bộ source kèm README/file môi trường và trọng số đã huấn luyện. Không yêu cầu slide. Web + Word chỉ là một phần bàn giao nếu thiếu source/trọng số; CLI + báo cáo + source/trọng số/môi trường có thể phù hợp yêu cầu, sau khi bổ sung các mục còn thiếu. Không thể bảo đảm giảng viên chấp nhận chỉ tiêu HR chưa đạt.

## Việc nên hoàn thiện theo thứ tự

1. Sửa README phản ánh kết quả thật, đường dẫn output và lệnh CPU; bỏ khuyến nghị làm slide như một bước bắt buộc trong review cũ. Đóng gói source, môi trường, decoder/checkpoint/proxy/bank/config và bảng/hình cần thiết.
2. Bổ sung Word từ artifact hiện có: median/failures, seed variability, vùng hỗ trợ từng fold, cặp ablation lambda=0, rise/cycle/tỷ lệ trích nhịp, hình đổi h/đổi z và ví dụ sai/không đo được. Có thể kèm CSV đầy đủ trong gói và dùng bảng rút gọn trong Word.
3. Tính histogram HR 20 bpm và thống kê loại theo lý do; bổ sung phân tầng motion từ ACC đúng mốc thời gian nếu truy cập được dữ liệu gốc. Không tự giả định valid = sạch/ít chuyển động.
4. Chạy CLI trên CPU máy thử, ghi CPU model/RAM/OS/runtime; đo decoder và toàn quy trình riêng, ghi RAM vận hành nếu đo. Không cần train lại để hoàn thiện bước này.
5. Đổi body Word 11,5 về 11 hoặc 12, render và kiểm tra lại 15–25 trang. Xác minh thông tin học kỳ/năm học: quy định ghi 2026–2027, đề cương ghi học kỳ II/2025–2026 và có nội dung giữa kỳ/cuối kỳ chưa thống nhất.
6. Nếu cần đạt mục tiêu HR thay vì chỉ hoàn thiện bàn giao: thiết kế vòng nghiên cứu mới kiểm chuẩn phép đo/proxy, ưu tiên đối chứng lambda=0 và khảo sát cải tiến trên validation. Test hiện tại đã được xem; không tiếp tục tối ưu rồi dùng chính test đó như bằng chứng độc lập mới.

Không cần chạy lại 40 lượt chỉ để lấy lại bảng/hình đã có. Không cần web, MCU hay các biến thể tùy chọn để bù cho thiếu sót trong thống kê và báo cáo.
