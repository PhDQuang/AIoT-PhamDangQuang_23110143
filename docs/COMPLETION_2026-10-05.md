# Hồ sơ hoàn thiện G2 ngày 05/10/2026

## Kết quả bàn giao

- Word: `reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx`, năm học **2026–2027**, thân bài Times New Roman 11 pt, đơn sắc, giãn dòng 1,2; giữ mục tiêu 15–25 trang. Bổ sung các bảng/hình còn thiếu từ kết quả thực nghiệm thật.
- ZIP: `delivery/PPG_CVAE_G2_Ban_giao.zip`: source, notebook, file môi trường, trọng số, cấu hình, manifest, bảng/hình, báo cáo và hướng dẫn chạy. Không chứa dữ liệu thô, credentials hoặc toàn bộ NPZ sinh lớn.
- `delivery/verification.json` và `MANIFEST_SHA256.csv` trong ZIP kiểm tra dung lượng, SHA256 và CRC. CLI CPU đã được chạy từ source giải nén ở thư mục tạm ngoài project.

## Các bổ sung thực hiện

| Hạng mục | File hoặc bằng chứng |
|---|---|
| Số người/cửa sổ trước–sau, HR min/max mỗi split/fold | `artifacts/completion/data_by_split.csv` |
| Histogram HR toàn miền, bin [a,a+20) | `hr_bins20.csv` và Hình 3 |
| Độ phủ và kiểm tra ngoài miền train mỗi target | `target_support.csv`, `generation_with_support.csv` |
| Lý do loại theo split và mức chuyển động | `exclusions_by_reason.csv`, `motion_by_split.csv`, Bảng 1a |
| ACC đồng bộ từng cửa sổ | `motion/motion_windows.csv`, `motion_definition.json`, `motion_thresholds_fold_*.json` |
| Bộ đo trên dữ liệu thật tổng hợp theo người | `real_validation_by_subject.csv`, `real_validation_subject_macro.csv`, Hình 6 |
| Median sai số và số thất bại | `median_and_failures.csv`, Bảng 3 và bảng 120 run gắn độ phủ |
| Giữ z đổi h, giữ h đổi z | Hình 11; dùng các index cố định từ NPZ đã đánh giá |
| Sai HR dù đo được và thất bại phép đo | Hình 9, `failure_example.json`; giữ bộ hình 20 mẫu trong results |
| Biến thiên giữa seed từng fold | Hình 10 và `results/summary/seed_summary_by_fold.csv` |
| Rise/cycle, tỷ lệ trích nhịp, biên độ riêng | Hình 14 và `morphology_comparison.csv` từng run |
| Tách tác động điều kiện khi lambda=0 | Định lượng no_condition so với no_hr_loss ở mục 5.8 |
| CPU model, RAM, OS/runtime, decoder và toàn quy trình | `local_cpu_environment.json`, `local_cpu_latency.csv`, Bảng 4 |
| Phiên bản thư viện local | `environment_local_versions.txt`; môi trường Kaggle giữ trong từng run |
| KL từng chiều | `kl_per_dimension_final_validation.csv`; mã train mới log `val_kl_dim_*` |
| README và lệnh đúng sau tải/giải nén | `README.md`, `HUONG_DAN_BAN_GIAO.md` trong ZIP |

Các file CSV bổ sung ở `artifacts/completion/`; trong ZIP tương ứng `completion/`.

## Thống kê chuyển động và nguồn

Job CPU `20261005-045644-ae0c99` đã COMPLETE và tải về thành công. Nó chỉ đọc bộ dữ liệu `ameersifat53/ppg-dalia-dataset`, không train hoặc đánh giá lại để chọn mô hình. Có 64.697 cửa sổ ACC; ghép một–một vào manifest của cả năm fold bằng subject/start/end.

Motion RMS là căn trung bình tổng bình phương ACC ba trục sau trừ trung bình từng trục trong 8 giây (256 mẫu ACC ở 32 Hz). Từng fold dùng phân vị 1/3 và 2/3 trên ứng viên train để mô tả low/medium/high; ngưỡng fold 1 khoảng 0,1685 và 0,4235 g. Đây là audit sau thực nghiệm, không phải ngưỡng chất lượng PPG và không làm thay tập hợp lệ.

Fold 1 gộp split: low 20.924 ứng viên, loại 4; medium 22.435, loại 6; high 21.338, loại 20. Tất cả 30 cửa sổ loại có lý do filter_boundary. Không kết luận chuyển động gây loại cửa sổ từ thống kê này. Từng fold/split và từng lý do được lưu riêng.

## Benchmark local

Máy thử Intel Core i5-1145G7, RAM 31,71 GiB, Windows 11, Python 3.12.15, PyTorch 2.14.1+cpu. Checkpoint fold 1 seed 42, FP32 CPU một luồng, batch 1, 100 decoder warmup, 1.000 đo mỗi HR.

| HR | Decoder median/P95, ms | Lấy z + decoder + đo HR median/P95, ms |
|---|---:|---:|
| 60 | 0,513 / 1,480 | 0,940 / 1,739 |
| 80 | 0,433 / 0,831 | 0,735 / 1,320 |
| 120 | 0,461 / 1,058 | 0,993 / 1,906 |

Working set khoảng 310,79 MiB và peak tiến trình 339,80 MiB: toàn Python/PyTorch cùng thư viện phân tích, không phải RAM riêng decoder. Không gồm loading, file I/O hoặc plotting trong latency. Không gộp số đo laptop vào benchmark Kaggle và không suy sang MCU.

## Kiểm tra và phần vẫn chưa đạt

- Toàn bộ 38 test qua; hai regression liên quan KL từng chiều và resource metadata được kiểm tra lại sau bổ sung assertion. Kiểm tra cú pháp source/notebook qua.
- Các bảng bin HR cộng lại khớp số cửa sổ hợp lệ; bảng run có 120 dòng và 120.000 mẫu, không trùng danh tính run; snapshot có 240 giá trị KL hữu hạn cho 15 checkpoint ×16 chiều.
- Report được render thành PDF/PNG và kiểm tra từng trang; QA trung gian không là tài liệu cần nộp.
- **Mục tiêu HR chưa đạt:** MAE cVAE 17,88/8,47/34,09 bpm tại 60/80/120. Không thay kết quả hoặc tuyên bố đã đạt ≤3 bpm.
- Không có lịch sử KL từng chiều của các epoch cũ; chỉ có snapshot checkpoint cuối. Không có ngưỡng vùng thưa được chốt trước, nên giữ số người/cửa sổ thay vì dựng nhãn hậu nghiệm.
- Chưa khảo sát MCU/INT8, downstream utility hoặc biến thể HR-prior. Các mục đó không bắt buộc để hoàn thiện hồ sơ hiện tại. CPU máy thử nằm trong phạm vi đề cương; không bắt buộc web hoặc slide.

Muốn cải thiện HR cần vòng nghiên cứu mới có đánh giá độc lập. Các bổ sung hiện tại hoàn thiện bằng chứng và bàn giao, không biến kết quả chưa đạt thành kết quả đạt mục tiêu.
