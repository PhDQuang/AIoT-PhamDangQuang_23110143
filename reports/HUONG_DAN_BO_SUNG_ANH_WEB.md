# Hướng dẫn bổ sung ảnh web vào báo cáo

Báo cáo mới: `Bao_cao_PPG_CVAE_AI_cho_IoT_Day_Du.docx`.

Ảnh giao diện chưa được chụp tự động vì Computer Use không xác minh được URL hiện tại của Chrome và đã dừng phiên thao tác. Hình kết quả backend thật tại ba HR đã được đưa vào mục **6.3**; hình này là biểu đồ suy luận, không phải ảnh chụp giao diện.

## Ảnh cần bổ sung ở mục 6.2

1. Khởi động `scripts/start_web.ps1`, mở http://127.0.0.1:8000 và tải lại trang bằng Ctrl+F5.
2. Chọn HR **80 bpm**, seed **42**, **1 cửa sổ**, bấm **Sinh PPG**.
3. Chụp toàn bộ giao diện, thấy phần điều kiện, ba ô HR, hai đường tín hiệu và bảng đánh giá. Nếu màn hình nhỏ, chụp riêng phần trên và phần bảng; đừng thu nhỏ đến mức chữ khó đọc.
4. Chèn vào đoạn “Vị trí bổ sung ảnh giao diện” trong mục 6.2; thay đoạn hướng dẫn đó bằng ảnh và chú thích **Giao diện PPG Studio sinh PPG theo HR và đo nhịp độc lập**.
5. Nếu muốn minh họa đủ ba HR, chụp thêm hai ảnh ở **60 bpm** và **120 bpm**, cùng seed 42 và một cửa sổ. Đặt sau ảnh 80 bpm hoặc trong phụ lục.
6. Cập nhật số thứ tự hình và mục lục trong Word: Ctrl+A → F9. Các ảnh có số đo riêng, không sửa HR đo để khớp HR mục tiêu.

## Ảnh bổ sung tùy chọn

- HR **80 bpm**, seed **42**, **20 cửa sổ**: chụp bảng MAE, V, A₃ và bộ chọn cửa sổ để minh họa sinh nhiều mẫu. Chỉ số này là lượt demo hiện tại, không thay kết quả năm fold.
- Giữ HR 80 bpm, thử seed 42 và 43: chụp hai dạng sóng để minh họa thay đổi latent.
- Mở CSV được tải từ web để minh họa các cột `window`, `time_seconds`, `cvae_ppg`, `gaussian_ppg`, `target_hr_bpm`, `seed`.

Không cần ảnh thiết bị MCU hoặc gateway cho phạm vi CPU máy thử của đề cương. Chỉ thêm ảnh thiết bị nếu đã thực hiện triển khai thực tế.
