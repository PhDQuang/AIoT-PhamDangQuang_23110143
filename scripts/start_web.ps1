$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& (Join-Path $projectRoot '.venv/Scripts/python.exe') -m ppg_cvae.web @args
