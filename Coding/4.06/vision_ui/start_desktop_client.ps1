$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$clientDir = Join-Path $root "client"
$nodeModules = Join-Path $clientDir "node_modules"

function Ensure-Backend {
    $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Host "Backend already listening on 127.0.0.1:8000" -ForegroundColor Green
        return
    }

    Write-Host "Starting FastAPI backend in a new PowerShell window..." -ForegroundColor Cyan
    Start-Process pwsh -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $root "start.ps1")
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
npm run desktop:dev
