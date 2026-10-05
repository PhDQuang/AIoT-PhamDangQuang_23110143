# Khôi phục khi job COMPLETE nhưng tải kết quả bị timeout

Job notebook 03 `20261004-065805-ebc5a4` đã hoàn tất trên Kaggle. Metadata ban đầu ghi `status: complete`, `download_status: error`. Các cảnh báo `SyntaxWarning` của mistune/nbconvert khi xuất HTML không làm notebook thất bại.

Bản KRun đang cài giới hạn mỗi tiến trình Kaggle CLI ở 120 giây, kể cả tải toàn bộ output. `--timeout` của `krun run`/`krun logs` chỉ điều chỉnh thời gian theo dõi job. Thông báo lỗi timeout dùng chung có nhắc đến submission dù lỗi đang xảy ra khi tải.

Thử tải lại job hiện có trước:

```powershell
krun output 20261004-065805-ebc5a4
```

Nếu vẫn timeout, từ thư mục project chạy trình tải với thời gian truyền 30 phút:

```powershell
& "$env:APPDATA/uv/tools/krun-cli/Scripts/python.exe" scripts/download_krun_output.py 20261004-065805-ebc5a4 --transfer-timeout 1800
```

Script dùng môi trường và thông tin đăng nhập KRun đã cài, tải vào `.krun/jobs/<job-id>/output`, cập nhật trạng thái tải qua KRun. Script không gửi kernel. Có thể thay job ID để tải kết quả của job khác. Nếu KRun được cài ở vị trí khác, dùng Python của môi trường chứa package `krun`.

Không cần chạy lại `krun run` cho lỗi tải output của job đã COMPLETE.
