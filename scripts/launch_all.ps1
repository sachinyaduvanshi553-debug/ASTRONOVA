param()
$env:PYTHONPATH = "C:\Users\sachi\OneDrive\Documents\ASTRONOVA"
Set-Location "C:\Users\sachi\OneDrive\Documents\ASTRONOVA"

$services = @(
    [PSCustomObject]@{ Name="Gateway";       Module="services.gateway.main:app";        Port=8000 },
    [PSCustomObject]@{ Name="Ingestion";     Module="services.ingestion.main:app";      Port=8001 },
    [PSCustomObject]@{ Name="Processing";    Module="services.processing.main:app";     Port=8002 },
    [PSCustomObject]@{ Name="Features";      Module="services.features.main:app";       Port=8003 },
    [PSCustomObject]@{ Name="Forecasting";   Module="services.forecasting.main:app";    Port=8004 },
    [PSCustomObject]@{ Name="CommsImpact";   Module="services.comms_impact.main:app";   Port=8005 },
    [PSCustomObject]@{ Name="EarthImpact";   Module="services.earth_impact.main:app";   Port=8006 },
    [PSCustomObject]@{ Name="SatelliteRisk"; Module="services.satellite_risk.main:app"; Port=8007 },
    [PSCustomObject]@{ Name="XAI";           Module="services.xai.main:app";            Port=8008 },
    [PSCustomObject]@{ Name="Notifications"; Module="services.notifications.main:app";  Port=8009 },
    [PSCustomObject]@{ Name="FlareCatalog";  Module="services.flare_catalog.main:app";  Port=8010 },
    [PSCustomObject]@{ Name="Copilot";       Module="services.copilot.main:app";        Port=8011 },
    [PSCustomObject]@{ Name="PhysicsEngine"; Module="services.physics_engine.main:app"; Port=8013 },
    [PSCustomObject]@{ Name="SolarVision";   Module="services.solar_vision.main:app";   Port=8014 }
)

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AstroNova: Starting All Backend Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

foreach ($svc in $services) {
    Write-Host ">> Starting $($svc.Name) on port $($svc.Port)..." -ForegroundColor Yellow
    $mod = $svc.Module
    $port = $svc.Port
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "set PYTHONPATH=C:\Users\sachi\OneDrive\Documents\ASTRONOVA && cd /d C:\Users\sachi\OneDrive\Documents\ASTRONOVA && uvicorn $mod --port $port --host 0.0.0.0 > logs\${port}.log 2>&1" `
        -WindowStyle Hidden
}

Write-Host ""
Write-Host "Waiting 8 seconds for all services to boot..." -ForegroundColor Gray
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Health Check Results" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($svc in $services) {
    $url = "http://127.0.0.1:$($svc.Port)/health"
    try {
        $null = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Write-Host "  OK  $($svc.Name.PadRight(16)) http://127.0.0.1:$($svc.Port)" -ForegroundColor Green
    } catch {
        $url2 = "http://127.0.0.1:$($svc.Port)/"
        try {
            $null = Invoke-WebRequest -Uri $url2 -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            Write-Host "  OK  $($svc.Name.PadRight(16)) http://127.0.0.1:$($svc.Port)" -ForegroundColor Green
        } catch {
            Write-Host "  ERR $($svc.Name.PadRight(16)) http://127.0.0.1:$($svc.Port) (may still boot)" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  All Service URLs" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
foreach ($svc in $services) {
    Write-Host "  http://127.0.0.1:$($svc.Port)  ->  $($svc.Name)" -ForegroundColor White
}
Write-Host ""
Write-Host "  Frontend:     http://localhost:3000" -ForegroundColor Magenta
Write-Host "  Inference UI: http://localhost:3000/image-inference" -ForegroundColor Magenta
Write-Host ""
