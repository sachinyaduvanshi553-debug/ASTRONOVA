$services = [ordered]@{
    "gateway" = 8000
    "ingestion" = 8001
    "processing" = 8002
    "features" = 8003
    "forecasting" = 8004
    "xai" = 8005
    "earth-impact" = 8006
    "satellite-risk" = 8007
    "rag" = 8008
    "copilot" = 8009
    "notifications" = 8010
}

$basePath = "c:\Users\sachi\OneDrive\Documents\ASTRONOVA"
$env:PYTHONPATH = "$basePath;$basePath\shared"

foreach ($svc in $services.GetEnumerator()) {
    $name = $svc.Name
    $port = $svc.Value
    Write-Host "Starting $name on port $port..."
    
    # Command to run in the new window
    $cmd = "cd '$basePath\services\$name'; if (Test-Path 'app\main.py') { uvicorn app.main:app --host 0.0.0.0 --port $port --reload } else { uvicorn main:app --host 0.0.0.0 --port $port --reload }"
    
    # Start a new PowerShell window minimized
    Start-Process powershell -ArgumentList "-NoExit","-Command", $cmd -WindowStyle Minimized
}

Write-Host "All backend services launched in minimized windows!"
