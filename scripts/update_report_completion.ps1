param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference='Stop'
$path=Join-Path $ProjectRoot 'reports/work/report_content.json'
$content=Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json
$out=Join-Path $ProjectRoot 'artifacts/completion'
$data=Import-Csv (Join-Path $out 'data_by_split.csv')
$macro=Import-Csv (Join-Path $out 'real_validation_subject_macro.csv')
$latency=Import-Csv (Join-Path $out 'local_cpu_latency.csv')
$environment=Get-Content (Join-Path $out 'local_cpu_environment.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$failures=Import-Csv (Join-Path $out 'median_and_failures.csv')
$culture=[Globalization.CultureInfo]::InvariantCulture
function N($value,[int]$digits=2) { ([double]::Parse([string]$value,$culture)).ToString("F$digits",[Globalization.CultureInfo]::GetCultureInfo('vi-VN')) }
function Page($title){$content | Where-Object title -eq $title}
$p=Page '2 1 Căn nhãn và tiền xử lý'
$p.table.headers=@('Fold','Train trước → sau','Val trước → sau','Test trước → sau')
$p.table.rows=@(1..5 | ForEach-Object {$f=$_; $r=@([string]$f); foreach($s in @('train','val','test')) {$g=$data | Where-Object { [int]$_.fold -eq $f -and $_.split -eq $s }; $r+= "$($g.windows_before) → $($g.windows_after)"}; ,$r})
$p.after=@('Bảng 1 Cửa sổ trước và sau xử lý. Mỗi fold có 64.697 ứng viên, 64.667 hợp lệ; train/val/test giữ 9/3/3 người. Có 30 cửa sổ bị loại vì filter_boundary. Bảng theo lý do và split được bàn giao trong completion/exclusions_by_reason.csv. Không cộng năm fold thành số quan sát độc lập.',
'Chuẩn hóa dùng các điểm train hợp lệ, tránh đếm lặp cửa sổ chồng lấp; val/test dùng thống kê train. Tập hợp lệ và mô hình đã khóa được giữ nguyên trong các phân tích bổ sung. Biên độ Z không có đơn vị điện áp cảm biến.')
$p=Page '2 2 Phân bố HR và độ phủ điều kiện'
$p.paragraphs=@('Hình 3 bổ sung histogram toàn miền với bin [a,a+20) bpm cho từng split và cả năm fold. completion/data_by_split.csv lưu HR min/max, số người và cửa sổ; hr_bins20.csv lưu số cửa sổ/người mỗi bin. target_support.csv ghi độ phủ ±10 bpm và trạng thái ngoài miền train cho từng fold, split và target.',
'Độ phủ được trình bày bằng số người/cửa sổ thực tế. Chưa có ngưỡng vùng thưa chốt trước thí nghiệm, nên không tạo một nhãn thưa tùy ý sau khi xem test. Các số liệu mô tả không được dùng để đổi detector hoặc chọn lại mô hình.')
$p.caption='Hình 3 Histogram HR bin 20 bpm của cả 5 fold. Mỗi đường là một fold; cửa sổ chồng lấp không độc lập.'
$p.after=@('Vùng quanh 60/80/120 bpm dùng [h−10,h+10). Bảng đầy đủ đi kèm generation_with_support.csv để đọc kết quả từng method/fold/seed cùng độ phủ train. HR trong miền train không bảo đảm mọi hình thái/hoạt động đều được hỗ trợ.',
'Phân tầng chuyển động dùng ACC cổ tay đồng bộ 32 Hz, chỉ phục vụ mô tả. Mức chuyển động không phải nhãn PPG sạch và không tham gia cVAE hoặc thay đổi việc loại cửa sổ.')
if(Test-Path (Join-Path $out 'motion_by_split.csv')) {
    $motion=Import-Csv (Join-Path $out 'motion_by_split.csv')
    $thresholds=Get-Content (Join-Path $out 'motion_thresholds_fold_01.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $rows=@(foreach($level in @('low','medium','high')) {
        $g=@($motion | Where-Object { [int]$_.fold -eq 1 -and $_.motion_level -eq $level })
        $before=($g | Measure-Object windows_before -Sum).Sum
        $after=($g | Measure-Object windows_after -Sum).Sum
        ,@($level,[string][int]$before,[string][int]$after,"$(N (100*($before-$after)/$before) 3)%")
    })
    $p | Add-Member -NotePropertyName table -NotePropertyValue @{headers=@('Mức chuyển động','Ứng viên','Hợp lệ','Tỷ lệ loại'); rows=$rows} -Force
    $p.after=@("Bảng 1a Chuyển động fold 1, gộp các split để mô tả 15 người. ACC RMS sau trừ trung bình từng trục trên 256 mẫu; ngưỡng low/medium/high là hai phân vị 1/3 và 2/3 của ứng viên train: $(N $thresholds.thresholds_g[0] 4) và $(N $thresholds.thresholds_g[1] 4) g. Mỗi fold có ngưỡng train riêng; audit thực hiện sau thực nghiệm, không thay tập hợp lệ.",
    'Bảng đầy đủ theo fold/split/mức chuyển động và lý do loại ở completion/motion_by_split.csv và exclusions_by_reason.csv. Có 64.697 cửa sổ ACC khớp manifest theo người/start/end. Mức chuyển động không phải nhãn PPG sạch; HR min/max, bin 20 bpm và độ phủ quanh target được bàn giao riêng.')
}
$p=Page '4 Huấn luyện và chọn cấu hình'
$p.after=@($p.after | Where-Object {$_ -notlike 'KL theo từng chiều được bổ sung*'})
$p.after+= 'KL theo từng chiều được bổ sung bằng snapshot trên toàn bộ validation tại 15 checkpoint cuối, trong completion/kl_per_dimension_final_validation.csv. Đây là phân tích sau thực nghiệm, không phải lịch sử epoch. Mã train đã bổ sung log từng chiều cho các lần chạy mới; không kết luận posterior collapse chỉ từ KL nhỏ.'
$p=Page '4 1 Độ tin cậy của phép đo HR'
$p.caption='Hình 6 MAE và V trên PPG val thật: tính theo từng người rồi lấy trung bình trong fold. Không phải chỉ số trên tín hiệu sinh.'
$proxyValues=@($macro | ForEach-Object {N $_.proxy_mae_bpm}) -join '; '
$peakValues=@($macro | ForEach-Object {N $_.peak_mae_bpm}) -join '; '
$p.after=@("MAE macro theo người ở fold 1–5: proxy $proxyValues bpm; detector $peakValues bpm. Bảng từng người và macro được bàn giao riêng. Sai số còn đáng kể so với mục tiêu 3 bpm; kết quả sinh là kết quả theo bộ đo đã khóa, chưa phải xác thực sinh lý.",
'Proxy cung cấp gradient và đã đóng băng; detector đo cuối không nhận HR mục tiêu. Proxy trả số hữu hạn không tương đương đạt chất lượng HR. Theo đề cương, khi proxy yếu cần ưu tiên đối chứng lambda=0 và báo giới hạn, không tăng độ phức tạp để che mẫu lỗi.')
$p=Page '5 Kết quả điều khiển HR'
$p.table.headers=@('HR','cVAE MAE ± SD','Gaussian MAE ± SD','Median cVAE / G','Lỗi đo cVAE / G')
for($i=0;$i -lt 3;$i++){ $h=@(60,80,120)[$i]; $cv=$failures | Where-Object {$_.method -eq 'cvae' -and [int]$_.target_hr -eq $h}; $g=$failures | Where-Object {$_.method -eq 'gaussian' -and [int]$_.target_hr -eq $h}; $p.table.rows[$i]=@($p.table.rows[$i][0],$p.table.rows[$i][1],$p.table.rows[$i][2],"$(N $cv.median_error_mean) / $(N $g.median_error_mean)","$($cv.failed_total) / $($g.failed_total)") }
$p.after=@('Bảng 3 MAE và trung bình median sai số từng run, đơn vị bpm. Lỗi đo là tổng trên 15.000 cửa sổ mỗi phương pháp/HR; đây không phải 15 quần thể độc lập. Trung bình MAE ba target: cVAE 20,15 bpm, Gaussian 3,32 bpm. generation_with_support.csv chứa đủ 120 dòng method/fold/seed/HR, MAE, median, V, A3, thất bại và độ phủ.',
'MAE Gaussian bằng 0 tại 120 chỉ áp dụng detector trên các mẫu đo được. Không suy ra chất lượng sinh lý hoàn hảo. cVAE chưa đạt MAE≤3 tại cả ba target; mục tiêu tài nguyên đạt không bù cho sai số này.')
$p=Page '5 1 Tỷ lệ đạt và khả năng đo được'
$p.caption2='Hình 9 V và hai ví dụ lỗi. cVAE index 0 tại 120 đo được nhưng sai HR; Gaussian là cửa sổ không đo được đầu tiên theo thứ tự fold/seed/index. Metadata nằm trong failure_example.json.'
$p.after=@('V cVAE gần 100% nhưng A3 thấp. Mẫu không đo được và mẫu đo được nhưng sai HR là hai trường hợp khác nhau; cả hai cần được hiển thị. Gaussian không đo được vẫn giữ trong mẫu số V/A3, không sinh bù. Bộ hình index 0..19 mỗi HR được giữ trong results của gói bàn giao.')
$p=Page '5 2 Phân bố HR của mẫu sinh'
$p.paragraphs=@('Hình 10 trình bày histogram HR của cVAE fold 1 seed 42 và MAE của từng seed ở cả 5 fold. Ba đường seed được giữ riêng để không che biến thiên bằng một giá trị trung bình. summary/seed_summary_by_fold.csv ghi mean, SD và số seed từng fold/target.')
$p.caption='Hình 10 Hàng trên: HR đo được của 1.000 cửa sổ mỗi target; nét đứng là HR yêu cầu. Hàng dưới: MAE từng seed 42/43/44 trong 5 fold.'
$p.after=@('Sai số thấp hơn ở 80 bpm không đủ chứng minh điều kiện đã được học tốt trên toàn miền. Mẫu cVAE ở 120 thường không tăng số chu kỳ tương ứng yêu cầu. Các đồ thị này mô tả phân bố đã đánh giá, không được dùng để chọn lại checkpoint hoặc điều chỉnh phép đo.',
'Chỉ số tổng hợp lấy trung bình seed trong fold rồi mean/SD qua fold. SD giữa seed được bàn giao riêng; không coi các cửa sổ chồng lấp hoặc 15 tổ hợp fold×seed là các quần thể độc lập.')
$p=Page '5 3 Minh họa tín hiệu sinh'
$p.paragraphs=@('Hình 11 tách hai phép thử: giữ z cố định và đổi HR ở cột trái; giữ HR=80 và đổi z ở cột phải. workflow.generate_fixed đặt lại cùng bộ sinh seed 42 tại mỗi HR, nên index 0 dùng cùng vector z. Các chỉ số 0,1,2 được chọn trước, không theo độ đẹp hay sai số.')
$p.caption='Hình 11 Đáp ứng điều kiện và latent của cVAE, fold 1 seed 42. Cột trái cùng z; cột phải cùng h. Mỗi đường là 8 giây PPG ở biên độ Z.'
$p.after=@('Đầu ra thay đổi khi đổi h hoặc z, nhưng thay đổi hình dạng chưa đảm bảo đúng số chu kỳ hay latent đã phân tách hoàn toàn. Mô hình vẫn sai HR lớn tại 120; các hình định tính phải đọc cùng MAE/A3 và phân bố tất cả cửa sổ.',
'Bộ hình 20 cửa sổ mỗi HR của cả cVAE và Gaussian có trong results. Ví dụ PPG thật/tái tạo nằm ở Hình 12; các mẫu thất bại nằm ở Hình 9. Không dùng một vài đường minh họa để thay thế đánh giá phân bố.')
$p=Page '5 6 Hình thái xung và biên độ'
$p.paragraphs=@('Hình 14 bổ sung rise/cycle và tỷ lệ trích nhịp bên cạnh rise, width50, biên độ SD và range. Tham chiếu test fold 1 cân bằng 100 cửa sổ mỗi người trong [h−10,h+10), tối đa 300 cửa sổ/HR; tín hiệu sinh có 1.000 cửa sổ/HR.')
$p.caption='Hình 14 Median và IQR đặc trưng tại fold 1 seed 42. Tỷ lệ trích nhịp tính trên ứng viên nhịp theo bộ trích, không phải tỷ lệ PPG sạch.'
$p.after=@('Rise median cVAE gần 0,313 s ở cả ba HR; dữ liệu thật giảm khoảng 0,359→0,227 s. Width50 cVAE 0,398→0,391→0,359 s so với thật 0,563→0,375→0,242 s. Biên độ Z được báo riêng để không che sai lệch bằng chuẩn hóa lại từng cửa sổ.',
'morphology_comparison.csv và test_reference_coverage.csv của mọi run chứa count, median/IQR, rise/cycle, tỷ lệ trích và độ phủ tham chiếu. Hình một fold là minh họa, không thay thế các bảng được bàn giao; trích được nhịp không xác nhận ý nghĩa sinh lý.')
$p=Page '5 8 Bóc tách thành phần cùng seed'
$p.after=@('So với no_hr_loss, cVAE giảm MAE 0,78 bpm ở 80 và 11,76 bpm ở 120, nhưng tăng 3,28 bpm ở 60. Đây là so sánh cùng seed 42 để khảo sát HR loss.',
'Tách riêng điều kiện khi lambda=0: MAE no_condition trừ no_hr_loss tại 60/80/120 là −0,37/−0,07/+1,14 bpm. Có điều kiện tốt hơn ở 120, nhưng không tốt hơn ở 60/80 trong kết quả này. Không có kiểm định ý nghĩa thống kê; cả hai chỉ một seed.',
'no_condition so với cVAE cơ sở đổi cả điều kiện và HR loss, nên không dùng cặp đó để quy tác động cho một thành phần. Chưa thực hiện HR-prior, beta=0 hoặc latent 8/32; đây là các biến thể tùy chọn.')
$p=Page '5 9 Tài nguyên decoder trên CPU'
$p.paragraphs=@('Decoder có 84.081 tham số, tệp 340.199 byte (0,340 MB); toàn cVAE 366.625 tham số. Benchmark Kaggle dùng FP32 CPU một luồng, batch 1, warmup 100 và 1.000 lượt đo/checkpoint; trung bình median/P95 là 0,319/0,382 ms. Hình 17 giữ số đo gốc, không trộn với máy local.',
"Bổ sung trên máy thử: $($environment.cpu), RAM $(N ([double]$environment.ram_bytes/1GB)) GiB; Windows 11, Python 3.12.15, PyTorch $($environment.torch). Thử checkpoint fold 1 seed 42, CPU FP32, một luồng, batch 1; 100 decoder warmup, 1.000 lượt mỗi target. Không gồm loading/I/O/vẽ hình.")
$p.table.headers=@('HR','Decoder median / P95 ms','Toàn quy trình median / P95 ms')
$p.table.rows=@($latency | ForEach-Object {,@($_.target_hr,"$(N $_.decoder_median_ms 3) / $(N $_.decoder_p95_ms 3)","$(N $_.pipeline_median_ms 3) / $(N $_.pipeline_p95_ms 3)")})
$p.after=@("Bảng 4 Đo local riêng decoder và lấy z + decoder + đo HR. Working set tiến trình khoảng $(N ([double]$environment.memory.working_set_bytes/1MB)) MiB, peak tiến trình $(N ([double]$environment.memory.process_peak_working_set_bytes/1MB)) MiB, gồm Python/PyTorch và thư viện phân tích; không phải RAM riêng decoder. File 0,340 MB không đại diện RAM vận hành. Chi tiết ở completion/local_cpu_environment.json.")
$p=Page '6 Khả năng tích hợp trong hệ thống IoT'
$p.after=@('CLI CPU đã chạy thật trên máy thử, lưu PPG, HR đo được và measurable cho 10 mẫu yêu cầu 80 bpm. Đầu ra demo ở completion/demo/generated_ppg.npz. Sinh chỉ dùng decoder, không cần encoder/proxy. HR đo được không bị thay bằng HR yêu cầu.',
'Đề cương cho phép CPU máy thử và chưa yêu cầu vi điều khiển. Web là giao diện tùy chọn; nếu làm phải hiển thị waveform, HR yêu cầu/đo và thất bại. Sơ đồ gateway chưa được triển khai phần cứng, không suy benchmark host sang MCU hoặc INT8.',
'Mô hình chưa đảm bảo HR đúng nên chỉ dùng như dữ liệu khảo sát có metadata nguồn sinh và HR đo, không thay cảm biến thật hay chứng nhận thiết bị. Các hướng cải tiến cần vòng validation/test độc lập; không tối ưu tiếp trên test đã xem.')
$p=Page '6 1 Kết luận và sản phẩm bàn giao'
$p.table.rows=@(@('Source và môi trường','src, scripts, notebooks, requirements, pyproject'),@('Trọng số và cấu hình','results/cvae, proxy, gaussian, ablations, tuning'),@('Bảng và hình','results/summary và các run; completion'),@('Báo cáo Word','reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx'),@('Kiểm chứng bàn giao','MANIFEST_SHA256.csv và HUONG_DAN_BAN_GIAO.md'))
$p.after=@('Bảng 5 Gói ZIP bàn giao lưu source, 25 checkpoint cVAE/ablation, 15 decoder export, proxy, bank Gaussian, cấu hình, bảng/hình và báo cáo. Dataset thô và các NPZ sinh lớn không được đóng lại; đường dẫn dataset/lệnh tái tạo được ghi kèm. Hash manifest kiểm tra nội dung gói.',
'Hoàn thiện hồ sơ không có nghĩa đạt mục tiêu HR: cVAE vẫn chưa đạt MAE≤3 bpm. Các phân tích bổ sung không thay checkpoint, detector hoặc tập cửa sổ đã khóa. Không yêu cầu slide; chưa có kết quả triển khai MCU hoặc downstream utility.')
[IO.File]::WriteAllText($path,($content | ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
Write-Output 'Report completion content updated'
