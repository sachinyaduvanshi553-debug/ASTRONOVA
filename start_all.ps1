$services = [ordered]@{
    "gateway" = 8000
    "ingestion" = 8001
    "processing" = 8002
    "features" = 8003
    "forecasting" = 8004
    "xai" = 8005
    "earth_impact" = 8006
    "satellite_risk" = 8007
    "rag" = 8008
    "copilot" = 8009
    "notifications" = 8010
}

$basePath = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$venvPython = "$basePath\venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) { $venvPython = "python" }

foreach ($svc in $services.GetEnumerator()) {
    $name = $svc.Name
    $port = $svc.Value
    Write-Host "Starting $name on port $port..."
    
    $cmd = "`$env:PYTHONPATH='$basePath;$basePath\shared'; cd '$basePath'; & '$venvPython' -m uvicorn services.$name.main:app --host 0.0.0.0 --port $port --reload"
    Start-Process powershell -ArgumentList "-NoExit","-Command", $cmd -WindowStyle Minimized
}

Write-Host "All backend services launched in minimized windows!"
