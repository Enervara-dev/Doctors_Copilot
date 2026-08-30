<#
.SYNOPSIS
    Starts the Clinical Consultation Copilot: backend API + frontend console.

.EXAMPLE
    .\run.ps1                      # production diarization, opens the console
    .\run.ps1 -Mode development    # no pyannote (mock speakers) - faster startup
    .\run.ps1 -ApiPort 8001 -NoBrowser
#>
param(
    [ValidateSet("production", "development")]
    [string]$Mode = "production",
    [int]$ApiPort = 8000,
    [int]$UiPort = 5500,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "No .venv found at $python. See README section 'Setup from scratch'."
}

# The ML stack (torch, pyannote, speechbrain) lives only in .venv. A system Python
# will import fastapi fine and then fail at startup when it loads the diarizer.
Write-Host "Checking environment..." -ForegroundColor Cyan
$check = & $python -c "import torch, pyannote.audio, fastapi, nest_asyncio; print('ok')" 2>&1
if ($check -notmatch "ok") {
    Write-Error "The virtualenv is missing dependencies:`n$check`nRun: .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

function Stop-Port([int]$Port) {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
Stop-Port $ApiPort
Stop-Port $UiPort

$env:CLINICAL_COPILOT_DIARIZATION_MODE = $Mode

Write-Host "Starting API on http://127.0.0.1:$ApiPort  (diarization: $Mode)" -ForegroundColor Cyan
$api = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $ApiPort `
    -WorkingDirectory $root -PassThru -NoNewWindow

# Production mode loads pyannote + ECAPA before it serves anything (~40 s cold).
Write-Host "Waiting for the models to load..." -NoNewline
$ready = $false
foreach ($i in 1..90) {
    Start-Sleep -Seconds 1
    try {
        Invoke-RestMethod "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 | Out-Null
        $ready = $true; break
    } catch { Write-Host "." -NoNewline }
}
Write-Host ""
if (-not $ready) {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    Write-Error "The API did not come up. Start it in the foreground to see why:`n  $python -m uvicorn app.main:app --port $ApiPort"
}
Write-Host "API ready." -ForegroundColor Green

Write-Host "Serving the console on http://127.0.0.1:$UiPort" -ForegroundColor Cyan
$ui = Start-Process -FilePath $python `
    -ArgumentList "-m", "http.server", $UiPort, "--bind", "127.0.0.1", "--directory", (Join-Path $root "frontend") `
    -WorkingDirectory $root -PassThru -NoNewWindow

$url = "http://127.0.0.1:$UiPort/?api=http://127.0.0.1:$ApiPort"
if (-not $NoBrowser) { Start-Process $url }

Write-Host ""
Write-Host "  Console : $url" -ForegroundColor Green
Write-Host "  API docs: http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both." -ForegroundColor DarkGray

try {
    Wait-Process -Id $api.Id
} finally {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $ui.Id  -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped." -ForegroundColor DarkGray
}
