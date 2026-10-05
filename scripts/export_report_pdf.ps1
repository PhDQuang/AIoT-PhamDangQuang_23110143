param([string]$ProjectRoot,[string]$ReportName='Bao_cao_PPG_CVAE_AI_cho_IoT')
$ErrorActionPreference='Stop'
$word=New-Object -ComObject Word.Application
$word.Visible=$false; $word.DisplayAlerts=0
try {
    $doc=$word.Documents.Open([string](Join-Path $ProjectRoot "reports/$ReportName.docx"))
    if($doc.TablesOfContents.Count -gt 0) { $doc.TablesOfContents.Item(1).Update() | Out-Null }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    if($doc.TablesOfContents.Count -gt 0) { $doc.TablesOfContents.Item(1).UpdatePageNumbers() | Out-Null }
    $doc.Save()
    [object]$pdfPath=[string](Join-Path $ProjectRoot "reports/work/$ReportName.pdf")
    [object]$pdfFormat=17
    $doc.SaveAs([ref]$pdfPath,[ref]$pdfFormat)
    Write-Output "PDF PAGES $($doc.ComputeStatistics(2))"
    $doc.Close([ref]0)
} finally {$word.Quit([ref]0)}
