param(
    [ValidateSet(2, 3, 4, 5)]
    [int[]]$Folds = @(2, 3, 4, 5)
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$krunPython = Join-Path $env:APPDATA 'uv/tools/krun-cli/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $krunPython)) {
    throw "KRun Python was not found: $krunPython"
}
Push-Location -LiteralPath $projectRoot
try {
    foreach ($fold in $Folds) {
        $foldName = 'fold_{0:D2}' -f $fold
        $notebook = "notebooks/03_tune_cvae_$foldName.ipynb"
        $proxyInput = "inputs/proxy/$foldName"
        if (-not (Test-Path -LiteralPath "$proxyInput/source_manifest.json")) {
            throw "Missing input package for $foldName"
        }
        Write-Host "Submitting tuning for $foldName..."
        $submission = & krun run $notebook --gpu T4 --dataset hcsinhgoethe/ppg-cvae-prepared-folds02-05 --input $proxyInput --cell-timeout 21600 --timeout 43200 --detach 2>&1
        $submitExit = $LASTEXITCODE
        $submission | ForEach-Object { Write-Host $_ }
        if ($submitExit -ne 0) {
            throw "Submission failed for $foldName. Inspect the existing job before retrying."
        }
        $submissionText = ($submission | ForEach-Object { $_.ToString() }) -join "`n"
        if ($submissionText -notmatch 'Job ID:\s*(\d{8}-\d{6}-[a-f0-9]{6})') {
            throw "Cannot find Job ID for $foldName. Inspect krun jobs; do not resubmit blindly."
        }
        $jobId = $Matches[1]
        Write-Host "Waiting for $foldName, job $jobId, then downloading output..."
        & $krunPython scripts/download_krun_output.py $jobId --wait --wait-timeout 43200 --transfer-timeout 3600
        if ($LASTEXITCODE -ne 0) {
            throw "Wait/download failed for $jobId. Resume that job before running the next fold."
        }
        Write-Host "Finished ${foldName}: $jobId"
    }
    Write-Host 'All requested tuning folds finished and downloaded. Review selected configurations before final training.'
}
finally {
    Pop-Location
}
