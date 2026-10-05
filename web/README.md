# PPG Studio

Web demo đúng nhiệm vụ G2: sinh một cửa sổ PPG **8 giây / 512 mẫu / 64 Hz** từ **HR mục tiêu và latent ngẫu nhiên**, không cần file đầu vào.

## Khởi động

```powershell
.\scripts\start_web.ps1
```

Mở http://127.0.0.1:8000. Dừng bằng Ctrl+C. Môi trường Python dùng các thư viện đã có trong requirements.txt; không cần thư viện web bổ sung.

Chọn HR 60/80/120 bpm, seed và 1/5/20 cửa sổ. Web dùng checkpoint cVAE fold 1 seed 42 và ngân hàng tham số Gaussian khớp trên train cùng fold. cVAE được vẽ xanh lá, Gaussian xanh dương. Seed cố định giúp tái tạo lượt sinh; đổi HR với cùng seed để khảo sát điều kiện hoặc đổi seed để khảo sát dạng sóng.

Giao diện hiển thị HR mục tiêu, HR đo độc lập, sai lệch, trạng thái đo được, MAE, V, A₃ và số cửa sổ không đo được. Chỉ số là của lượt demo hiện tại; bảng kết quả chính của báo cáo dùng năm fold và ba seed. Mỗi cửa sổ là độc lập, không ghép thành một chuỗi liên tục. Hai phương pháp không được đánh giá bằng RMSE giữa hai waveform ngẫu nhiên.

Tải PNG cho cửa sổ đang xem; CSV cho tất cả cửa sổ với thời gian tương đối; JSON lưu cả tín hiệu, cấu hình và số đo. Biên độ là đơn vị chuẩn hóa của mô hình, không phải điện áp cảm biến.

API chính: `POST /api/synthesize` với JSON `{"hr":80,"seed":42,"count":1}`. API đọc file cũ được giữ để tương thích các thử nghiệm bổ sung; giao diện chính không còn luồng dự báo 8s → 16s.

Checkpoint được tìm trong `results`, output job cuối và `artifacts`. Có thể truyền `--checkpoint` và `--detector-config` qua script khởi động. Demo mặc định chỉ lắng nghe trên localhost.

Hướng dẫn ảnh báo cáo: `reports/HUONG_DAN_BO_SUNG_ANH_WEB.md`.
