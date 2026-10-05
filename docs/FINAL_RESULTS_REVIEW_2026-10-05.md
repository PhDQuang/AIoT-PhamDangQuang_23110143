# Kiểm tra kết quả chạy xuyên đêm

Job `20261004-151954-483f31`: Kaggle COMPLETE, download COMPLETE, không có error. Pipeline chạy 90,81 phút. Các bước thực nghiệm 1–4 đã hoàn tất; báo cáo chưa được tạo bởi job này.

## Kiểm tra artifact

- Đủ 5 fold và 40 method/fold/seed run: cVAE và Gaussian mỗi loại 15, no_hr_loss và no_condition mỗi loại 5.
- Đủ 120 dòng generation summary, 1000 mẫu/HR tại 60/80/120 cho mỗi run; 120.000 dòng sample metrics.
- Đã mở thành công 25 checkpoint mô hình (15 cVAE và 10 ablation).
- Đã kiểm tra CRC và đọc 120 NPZ tín hiệu sinh: shape (1000, 1, 512), không NaN/Inf.
- Đủ 25 bảng paired test reconstruction, RMSE hữu hạn.
- Đủ 15 bản CPU resource metrics và decoder export.
- Không có file artifact rỗng ngoài placeholder .gitkeep.
- Hash checkpoint/bank Gaussian, detector và normalization khớp snapshot protocol đã lưu trước test.
- Tính lại phép tổng hợp seed trước, fold sau khớp cross_fold_summary.csv.
- Đã xem hình cross_fold_generation.png: vẽ đủ 4 phương pháp, error bar là độ lệch chuẩn giữa fold, không phải khoảng tin cậy. Trục V đang zoom 0,97–1,00; cần ghi rõ khi đưa vào báo cáo.

## Kết quả HR

Trung bình đều giữa 3 target HR, sau khi trung bình seed trong fold rồi giữa 5 fold:

| Phương pháp | MAE bpm | A3 |
|---|---:|---:|
| cVAE | 20,15 | 9,54% |
| Two-Gaussian | 3,32 | 94,26% |
| no_hr_loss | 22,85 | 10,13% |
| no_condition | 23,09 | 10,60% |

cVAE và Gaussian dùng 3 seed; ablation chỉ seed 42 theo protocol. Không dùng bảng trung bình khác số seed này để suy luận trực tiếp tác động causal của ablation; so sánh tác động với cVAE seed 42 khi viết báo cáo.

| Target HR | cVAE MAE bpm | cVAE A3 | Gaussian MAE bpm | Gaussian A3 |
|---|---:|---:|---:|---:|
| 60 | 17,88 | 5,53% | 7,07 | 88,09% |
| 80 | 8,47 | 21,15% | 2,90 | 95,30% |
| 120 | 34,09 | 1,95% | 0,00 | 99,39% |

cVAE có V gần 100% nhưng điều đó chỉ cho thấy detector trả được HR, không chứng minh HR đúng. cVAE chưa đạt khả năng điều khiển HR tốt, đặc biệt tại 120 bpm. Gaussian vượt cVAE theo bộ đo HR đang dùng; chưa suy ra Gaussian tốt hơn mọi mặt về hình thái hoặc độ giống PPG thật. MAE Gaussian=0 tại 120 bpm là kết quả theo detector trên các mẫu đo được, không phải bằng chứng mô phỏng sinh lý hoàn hảo. Proxy/detector có sai số real validation đã ghi trong các review trước; cần nêu giới hạn này trong báo cáo.

## Cấu hình chọn và tài nguyên

Fold 1/3/4/5 chọn beta=0.01, lambda_hr=0.1. Fold 2 chọn beta=0.1, lambda_hr=1.0.

Decoder state_dict: 340.199 byte (~0,340 MB). Trung bình median latency của 15 checkpoint khoảng 0,319 ms; trung bình P95 khoảng 0,382 ms. Benchmark FP32, CPU Kaggle, 1 thread, batch 1, 100 warmup/1000 lần đo, loại loading/I/O/plot. Không dùng số này để khẳng định latency trên laptop, MCU hoặc INT8.

## Việc tiếp theo

Không cần chạy lại training/evaluation để hoàn tất artifact hiện tại. Chuẩn bị báo cáo và phân tích waveform/morphology/reconstruction từ các bảng đã có. Kết luận cần phản ánh cVAE thua baseline về độ chính xác HR, thay vì tuyên bố mô hình tốt hơn. Test đã được sử dụng trong đánh giá cuối; giữ nguyên các quyết định protocol khi trình bày kết quả.

Artifact gốc: `.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts/`.
File chính: `summary/cross_fold_summary.csv`, `summary/resource_metrics.csv`, `figures/cross_fold_generation.png`, `overnight/locked_evaluation_protocol.json`, `overnight/progress.json`.
