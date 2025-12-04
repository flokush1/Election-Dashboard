# Restart Flask Server with Debug Output
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   RESTARTING FLASK SERVER" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Kill any existing Python processes running model_api.py
Write-Host "Checking for existing Flask server processes..." -ForegroundColor White
$processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -like "*model_api*" -or 
    (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue).OwningProcess -contains $_.Id
}

if ($processes) {
    Write-Host "Found $($processes.Count) process(es) to terminate" -ForegroundColor Yellow
    $processes | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "✅ Existing processes terminated" -ForegroundColor Green
} else {
    Write-Host "No existing Flask server found" -ForegroundColor Gray
}

# Start the Flask server
Write-Host "`nStarting Flask server on port 5000..." -ForegroundColor White
Write-Host "Watch this console for debug output!`n" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

python model_api.py
