$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$clientDir = Join-Path $root "client"
$nodeModules = Join-Path $clientDir "node_modules"
$backendPort = 8765

function Ensure-Backend {
    $listener = Get-NetTCPConnection -LocalPort $backendPort -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Host "Backend already listening on 127.0.0.1:$backendPort" -ForegroundColor Green
        return
    }

    Write-Host "Starting FastAPI backend in a new PowerShell window..." -ForegroundColor Cyan
    Start-Process pwsh -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $root "start.ps1"),
        "-Port",
        "$backendPort"
    ) | Out-Null

    Start-Sleep -Seconds 2
}

function Ensure-NodeModules {
    if (Test-Path $nodeModules) {
        return
    }

    Write-Host "Installing client dependencies..." -ForegroundColor Cyan
    Push-Location $clientDir
    try {
        npm install
    } finally {
        Pop-Location
    }
}

Ensure-Backend
Ensure-NodeModules

Set-Location $clientDir
$env:VITE_API_BASE = "http://127.0.0.1:$backendPort"
npm run desktop:dev
