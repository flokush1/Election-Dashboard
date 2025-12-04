# Backend Diagnostic Test Script
# Run this to verify backend is working correctly

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   BACKEND DIAGNOSTIC TEST SUITE" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: Check if backend is running
Write-Host "Test 1: Checking if backend is running..." -ForegroundColor White
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -Method GET -TimeoutSec 5
    $health = $response.Content | ConvertFrom-Json
    
    Write-Host "✅ Backend is RUNNING" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor Gray
    Write-Host "   Model Loaded: $($health.model_loaded)" -ForegroundColor Gray
    Write-Host "   Uploaded Voters: $($health.data_status.uploaded_voters_count)" -ForegroundColor Gray
    
    if ($health.data_status.uploaded_voters_count -gt 0) {
        Write-Host "   Sample IDs: $($health.data_status.sample_voter_ids -join ', ')" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "❌ Backend is NOT RUNNING" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host "`n   To start backend, run: python model_api.py" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 2: Check if frontend is running
Write-Host "Test 2: Checking if frontend is running..." -ForegroundColor White
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:3000" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ Frontend is RUNNING at http://localhost:3000" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Frontend may not be running" -ForegroundColor Yellow
    Write-Host "   To start frontend, run: npm run dev" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Check Python process
Write-Host "Test 3: Checking Python process..." -ForegroundColor White
$pythonProcess = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcess) {
    Write-Host "✅ Python process found" -ForegroundColor Green
    $pythonProcess | Select-Object Id, ProcessName, StartTime | Format-Table -AutoSize
} else {
    Write-Host "❌ No Python process found" -ForegroundColor Red
}

Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   DIAGNOSTIC SUMMARY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

if ($health.data_status.uploaded_voters_count -gt 0) {
    Write-Host "✅ Everything looks good!" -ForegroundColor Green
    Write-Host "   Backend has $($health.data_status.uploaded_voters_count) voters loaded." -ForegroundColor Green
    Write-Host "`n   You can search for these voter IDs:" -ForegroundColor White
    foreach ($id in $health.data_status.sample_voter_ids) {
        Write-Host "      - $id" -ForegroundColor Cyan
    }
} else {
    Write-Host "⚠️ Backend is running but NO DATA uploaded yet" -ForegroundColor Yellow
    Write-Host "`n   Next steps:" -ForegroundColor White
    Write-Host "   1. Go to http://localhost:3000" -ForegroundColor Cyan
    Write-Host "   2. Navigate to 'Electoral Individual Voter Prediction Dashboard'" -ForegroundColor Cyan
    Write-Host "   3. Upload your voter Excel file" -ForegroundColor Cyan
    Write-Host "   4. Wait for success message" -ForegroundColor Cyan
    Write-Host "   5. Search for voter IDs" -ForegroundColor Cyan
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
