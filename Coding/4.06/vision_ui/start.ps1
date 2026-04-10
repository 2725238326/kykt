param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Expected: $python"
}

Set-Location $root

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $pidList = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    Write-Host "Port $Port is already in use by PID(s): $pidList" -ForegroundColor Yellow
    Write-Host "Close the old KYKT Vision UI terminal first, or stop that process, then run start.ps1 again." -ForegroundColor Yellow
    exit 1
}

& $python -m uvicorn app:app --host 127.0.0.1 --port $Port
