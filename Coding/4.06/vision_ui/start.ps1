$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Expected: $python"
}

Set-Location $root
& $python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
