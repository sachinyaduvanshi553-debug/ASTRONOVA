# ============================================================
# AstroNova - Full Local Dev Launcher
# Launches: Gateway (8000), Forecasting (8004), Copilot (8009), Frontend (3000)
# ============================================================

$BaseDir = "c:\Users\sachi\OneDrive\Documents\ASTRONOVA"
$VenvPython = "$BaseDir\venv\Scripts\python.exe"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  AstroNova Local Dev Launcher" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# -- 1. Gateway (port 8000) ----------------------------------
Write-Host "[1/4] Starting API Gateway on http://localhost:8000 ..." -ForegroundColor Yellow
$gwCmd = "& '$VenvPython' -m uvicorn services.gateway.main:app --host 0.0.0.0 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "`$env:PYTHONPATH='$BaseDir;$BaseDir\shared'; cd '$BaseDir'; $gwCmd"

Start-Sleep -Seconds 2

# -- 2. Forecasting Service (port 8004) ---------------------
Write-Host "[2/4] Starting Forecasting Service on http://localhost:8004 ..." -ForegroundColor Yellow
$fcCmd = "& '$VenvPython' -m uvicorn services.forecasting.main:app --host 0.0.0.0 --port 8004 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "`$env:PYTHONPATH='$BaseDir;$BaseDir\shared'; cd '$BaseDir'; $fcCmd"

Start-Sleep -Seconds 2

# -- 3. Copilot Service (port 8009) ------------------------
Write-Host "[3/4] Starting Copilot Service on http://localhost:8009 ..." -ForegroundColor Yellow
$cpCmd = "& '$VenvPython' -m uvicorn services.copilot.main:app --host 0.0.0.0 --port 8009 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "`$env:PYTHONPATH='$BaseDir;$BaseDir\shared'; cd '$BaseDir'; $cpCmd"

Start-Sleep -Seconds 2

# -- 4. Frontend (Next.js on port 3000) ---------------------
Write-Host "[4/4] Starting Frontend (Next.js) on http://localhost:3000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$BaseDir\frontend'; npm run dev"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  All services launched!" -ForegroundColor Green
Write-Host "-----------------------------------------" -ForegroundColor Green
Write-Host "  Frontend UI   : http://localhost:3000" -ForegroundColor White
Write-Host "  API Gateway   : http://localhost:8000" -ForegroundColor White
Write-Host "  Gateway Docs  : http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Forecasting   : http://localhost:8004" -ForegroundColor White
Write-Host "  Forecast Docs : http://localhost:8004/docs" -ForegroundColor White
Write-Host "  Copilot API   : http://localhost:8009" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Close individual PowerShell windows to stop services." -ForegroundColor Gray
