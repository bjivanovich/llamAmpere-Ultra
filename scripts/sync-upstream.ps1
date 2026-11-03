# llamAmpere-Ultra: One-Click Upstream Synchronizer
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  llamAmpere-Ultra: Checking Ecosystem & Upstream Updates" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

$pythonExe = "d:\bjivanovich2017\My Drive\aibot\python\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe "$PSScriptRoot\sync_upstream.py"
