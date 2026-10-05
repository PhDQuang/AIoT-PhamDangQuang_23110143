param([string]$ProjectRoot,[string]$ReportName='Bao_cao_PPG_CVAE_AI_cho_IoT',[string]$RenderFolder='render')
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
[void][Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime]
[void][Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
[void][Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime]
$genericAsTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await-Result($Operation,[Type]$ResultType) {
    $task=$genericAsTask.MakeGenericMethod($ResultType).Invoke($null,@($Operation))
    $task.GetAwaiter().GetResult()
}
function Await-Action($Operation) {
    $method=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and -not $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' } | Select-Object -First 1
    $task=$method.Invoke($null,@($Operation)); [void]$task.GetAwaiter().GetResult()
}
$pdfFile=Await-Result ([Windows.Storage.StorageFile]::GetFileFromPathAsync((Join-Path $ProjectRoot "reports/work/$ReportName.pdf"))) ([Windows.Storage.StorageFile])
$pdfDoc=Await-Result ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($pdfFile)) ([Windows.Data.Pdf.PdfDocument])
$out=Join-Path $ProjectRoot "reports/work/$RenderFolder"; [void][IO.Directory]::CreateDirectory($out)
for($i=0;$i -lt $pdfDoc.PageCount;$i++) {
    $page=$pdfDoc.GetPage([uint32]$i)
    $stream=New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    $options=New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $options.DestinationWidth=1240; $options.DestinationHeight=1754
    Await-Action ($page.RenderToStreamAsync($stream,$options))
    $reader=New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    [void](Await-Result ($reader.LoadAsync([uint32]$stream.Size)) ([uint32]))
    $bytes=New-Object byte[] ([int]$stream.Size); $reader.ReadBytes($bytes)
    [IO.File]::WriteAllBytes((Join-Path $out ('page-{0:d2}.png' -f ($i+1))),$bytes)
    $reader.Dispose(); $stream.Dispose(); $page.Dispose()
}
Write-Output "RENDERED $($pdfDoc.PageCount) pages"
