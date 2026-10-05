param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
$content = Get-Content (Join-Path $ProjectRoot 'reports/work/report_content.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$output = Join-Path $ProjectRoot 'reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx'
$pdf = Join-Path $ProjectRoot 'reports/work/Bao_cao_PPG_CVAE_AI_cho_IoT.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Add()
    $setup = $doc.PageSetup
    $setup.PaperSize = 7
    $setup.TopMargin = $word.CentimetersToPoints(2)
    $setup.BottomMargin = $word.CentimetersToPoints(2)
    $setup.LeftMargin = $word.CentimetersToPoints(2.5)
    $setup.RightMargin = $word.CentimetersToPoints(2)
    $setup.HeaderDistance = $word.CentimetersToPoints(0.8)
    $setup.FooterDistance = $word.CentimetersToPoints(0.8)
    $setup.DifferentFirstPageHeaderFooter = -1
    $normal = $doc.Styles.Item(-1)
    $normal.Font.Name = 'Times New Roman'; $normal.Font.Size = 11; $normal.Font.Color = 0
    $normal.ParagraphFormat.Alignment = 3
    $normal.ParagraphFormat.SpaceAfter = 6
    $normal.ParagraphFormat.LineSpacingRule = 5
    $normal.ParagraphFormat.LineSpacing = 13.2
    $normal.ParagraphFormat.WidowControl = -1
    foreach ($styleId in @(-2,-3,-63)) {
        $st = $doc.Styles.Item($styleId)
        $st.Font.Name = 'Times New Roman'; $st.Font.Color = 0
        $st.ParagraphFormat.Borders.Enable = 0
    }
    $doc.Styles.Item(-2).Font.Size = 15
    $doc.Styles.Item(-2).Font.Bold = -1
    $doc.Styles.Item(-2).ParagraphFormat.SpaceBefore = 0
    $doc.Styles.Item(-2).ParagraphFormat.SpaceAfter = 12
    $doc.Styles.Item(-2).ParagraphFormat.KeepWithNext = -1
    $doc.Styles.Item(-63).Font.Size = 20
    $doc.Styles.Item(-63).ParagraphFormat.Alignment = 1
    $doc.Styles.Item(-63).ParagraphFormat.SpaceAfter = 18
    $header = $doc.Sections.Item(1).Headers.Item(1).Range
    $header.Text = 'PHẠM ĐĂNG QUANG   |   TRÍ TUỆ NHÂN TẠO CHO IOT'
    $header.Font.Name='Times New Roman'; $header.Font.Size=9; $header.Font.Color=0
    $header.ParagraphFormat.Alignment=2
    $header.ParagraphFormat.Borders.Enable=0
    $footer = $doc.Sections.Item(1).Footers.Item(1).Range
    $footer.Font.Name='Times New Roman'; $footer.Font.Size=10
    $footer.ParagraphFormat.Alignment=1
    [void]$doc.Sections.Item(1).Footers.Item(1).PageNumbers.Add(1,$false)
    $sel = $word.Selection
    function Add-Paragraph([string]$Text,[double]$Size=11,[int]$Align=3,[bool]$Bold=$false) {
        $sel.Style = $doc.Styles.Item(-1)
        $sel.Font.Name='Times New Roman'; $sel.Font.Size=$Size; $sel.Font.Bold=[int]$Bold * -1
        $sel.Font.Italic=0; $sel.Font.Color=0
        $sel.ParagraphFormat.Alignment=$Align
        $sel.ParagraphFormat.KeepWithNext=0
        $sel.ParagraphFormat.SpaceAfter=6
        $sel.ParagraphFormat.LineSpacingRule=5
        $sel.ParagraphFormat.LineSpacing=$Size*1.2
        $sel.TypeText($Text); $sel.TypeParagraph()
    }
    function Add-Figure([string]$Name,[string]$Caption) {
        $sel.Style = $doc.Styles.Item(-1)
        $sel.ParagraphFormat.Alignment=1
        $sel.ParagraphFormat.KeepWithNext=-1
        $sel.ParagraphFormat.SpaceAfter=3
        $picture = $sel.InlineShapes.AddPicture((Join-Path $ProjectRoot "reports/work/figures/$Name.png"),$false,$true)
        $picture.LockAspectRatio=-1; $picture.Width=462
        $sel.SetRange($picture.Range.End,$picture.Range.End)
        $sel.TypeParagraph()
        Add-Paragraph $Caption 11 1
    }
    function Add-Table($Data) {
        $sel.Style=$doc.Styles.Item(-1)
        $tab=$doc.Tables.Add($sel.Range,($Data.rows.Count+1),$Data.headers.Count)
        $tab.AllowAutoFit=$false
        $tab.PreferredWidthType=3; $tab.PreferredWidth=468
        $tab.TopPadding=5; $tab.BottomPadding=5; $tab.LeftPadding=6; $tab.RightPadding=6
        $tab.Range.Font.Name='Times New Roman'; $tab.Range.Font.Size=11; $tab.Range.Font.Bold=0
        $tab.Range.ParagraphFormat.Alignment=0
        $tab.Range.ParagraphFormat.SpaceAfter=0
        $tab.Range.ParagraphFormat.LineSpacingRule=0
        $tab.Range.Cells.VerticalAlignment=1
        $tab.Rows.AllowBreakAcrossPages=$false
        $tab.Rows.Item(1).HeadingFormat=-1
        $tab.Borders.Enable=1
        foreach($border in $tab.Borders) {$border.LineStyle=1; $border.LineWidth=4; $border.Color=14277081}
        $tab.Borders.Item(-7).LineStyle=0
        $tab.Borders.Item(-8).LineStyle=0
        if($Data.headers.Count -eq 2) { $tab.Columns.Item(1).Width=155; $tab.Columns.Item(2).Width=313 }
        elseif($Data.headers.Count -eq 3) {$tab.Columns.Item(1).Width=122; $tab.Columns.Item(2).Width=173; $tab.Columns.Item(3).Width=173}
        else {for($j=1;$j -le $Data.headers.Count;$j++){$tab.Columns.Item($j).Width=[single](468/$Data.headers.Count)}}
        for($j=0;$j -lt $Data.headers.Count;$j++) {
            $cell=$tab.Cell(1,$j+1); $cell.Range.Text=[string]$Data.headers[$j]
            $cell.Shading.BackgroundPatternColor=0
            $cell.Range.Font.Color=16777215; $cell.Range.Font.Bold=-1
        }
        for($i=0;$i -lt $Data.rows.Count;$i++) {
            for($j=0;$j -lt $Data.headers.Count;$j++) {
                $cell=$tab.Cell($i+2,$j+1); $cell.Range.Text=[string]$Data.rows[$i][$j]
                $cell.Range.Font.Color=0
                if($i%2 -eq 1){$cell.Shading.BackgroundPatternColor=15921906}
            }
        }
        $sel.SetRange($tab.Range.End,$tab.Range.End)
        $sel.TypeParagraph()
    }
    function Add-Equation([string]$Text) {
        $sel.Style=$doc.Styles.Item(-1); $sel.Font.Name='Cambria Math'; $sel.Font.Size=12
        $sel.ParagraphFormat.Alignment=1; $sel.ParagraphFormat.SpaceAfter=10
        $start=$sel.Start; $sel.TypeText($Text); $end=$sel.End
        $range=$doc.Range($start,$end)
        [void]$doc.OMaths.Add($range)
        $doc.OMaths.Item($doc.OMaths.Count).Type=0
        $doc.OMaths.Item($doc.OMaths.Count).BuildUp()
        $sel.SetRange($range.End,$range.End); $sel.TypeParagraph()
    }
    Add-Paragraph 'BỘ GIÁO DỤC VÀ ĐÀO TẠO' 13 1 $true
    Add-Paragraph 'TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP HCM' 13 1 $true
    Add-Paragraph 'KHOA CÔNG NGHỆ THÔNG TIN' 13 1 $true
    Add-Paragraph '' 12 1
    Add-Paragraph '' 12 1
    Add-Paragraph 'BÁO CÁO TIỂU LUẬN CUỐI KHÓA' 17 1 $true
    Add-Paragraph 'HỌC PHẦN TRÍ TUỆ NHÂN TẠO CHO IOT' 14 1 $true
    Add-Paragraph '' 12 1
    $sel.Style=$doc.Styles.Item(-63)
    $sel.Font.Size=20; $sel.Font.Bold=-1
    $sel.ParagraphFormat.Alignment=1
    $sel.ParagraphFormat.LineSpacing=24
    $sel.TypeText('Sinh tín hiệu PPG theo nhịp tim bằng mạng tự mã hóa biến phân có điều kiện')
    $sel.TypeParagraph()
    Add-Paragraph 'Đánh giá trên PPG DaLiA với 5 fold theo người' 12 1
    Add-Paragraph '' 12 1
    Add-Paragraph 'Giảng viên hướng dẫn   ThS Hồ Nhựt Minh' 13 0
    Add-Paragraph 'Sinh viên thực hiện       Phạm Đăng Quang' 13 0
    Add-Paragraph 'Mã số sinh viên              23110143' 13 0
    Add-Paragraph 'Lớp học phần                 AIOT331185_01CLC' 13 0
    Add-Paragraph 'Mã đề tài                       G2' 13 0
    Add-Paragraph 'Năm học                          2026–2027' 13 0
    Add-Paragraph '' 12 1
    Add-Paragraph 'TP Hồ Chí Minh tháng 10 năm 2026' 12 1
    foreach($page in $content) {
        Write-Output "AUTHOR $($page.title)"
        $sel.InsertBreak(7)
        $sel.Style=$doc.Styles.Item(-2)
        $sel.Font.Name='Times New Roman'; $sel.Font.Size=15
        $sel.ParagraphFormat.SpaceAfter=12
        $sel.ParagraphFormat.LineSpacing=18
        $sel.ParagraphFormat.KeepWithNext=-1
        $sel.Font.Color=0; $sel.Font.Italic=0; $sel.Font.Bold=-1
        $sel.ParagraphFormat.Alignment=0
        $sel.TypeText($page.title); $sel.TypeParagraph()
        $size=11
        foreach($text in $page.paragraphs) { Add-Paragraph $text $size }
        if($page.equations){foreach($eq in $page.equations){Add-Equation $eq}}
        if($page.figure){Add-Figure $page.figure $page.caption}
        if($page.figure2){Add-Figure $page.figure2 $page.caption2}
        if($page.table){Add-Table $page.table}
        foreach($text in $page.after){Add-Paragraph $text $size}
    }
    Write-Output 'PAGINATE'
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    Write-Output 'SAVE DOCX'
    [object]$savePath=[string]$output
    [object]$saveFormat=16
    $doc.SaveAs([ref]$savePath,[ref]$saveFormat)
    Write-Output "PAGES $($doc.ComputeStatistics(2))"
    Write-Output "EQUATIONS $($doc.OMaths.Count)"
    Write-Output "IMAGES $($doc.InlineShapes.Count)"
    $doc.Close([ref]0)
} finally {$word.Quit([ref]0)}
$finalizer=[ScriptBlock]::Create((Get-Content (Join-Path $ProjectRoot 'scripts/finalize_report_ooxml.ps1') -Raw -Encoding UTF8))
& $finalizer -ProjectRoot $ProjectRoot
$exporter=[ScriptBlock]::Create((Get-Content (Join-Path $ProjectRoot 'scripts/export_report_pdf.ps1') -Raw -Encoding UTF8))
& $exporter -ProjectRoot $ProjectRoot
