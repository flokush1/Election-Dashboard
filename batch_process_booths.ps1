# Batch Process Multiple Booths
# This script processes multiple booths at once

$booths = @(
    @{
        Booth = 103
        Block = "E"
        Locality = "B.K.Dutt Colony"
        GeoJSON = "Block-E, B.K.Dutt Colony.geojson"
        Assembly = "New Delhi"
    }
    # Add more booths here:
    # @{
    #     Booth = 104
    #     Block = "C"
    #     Locality = "B.K.Dutt Colony"
    #     GeoJSON = "Block-C, B.K.Dutt Colony.geojson"
    #     Assembly = "New Delhi"
    # }
)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "BATCH PROCESSING $($booths.Count) BOOTH(S)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan

$successCount = 0
$failCount = 0
$results = @()

foreach ($b in $booths) {
    Write-Host "`nProcessing Booth $($b.Booth)..." -ForegroundColor Yellow
    
    $cmd = "python integrate_booth_predictions.py --booth $($b.Booth) --assembly `"$($b.Assembly)`" --locality `"$($b.Locality)`" --geojson `"$($b.GeoJSON)`""
    
    if ($b.Block) {
        $cmd += " --block `"$($b.Block)`""
    }
    
    try {
        Invoke-Expression $cmd
        
        if ($LASTEXITCODE -eq 0) {
            $successCount++
            $results += @{
                Booth = $b.Booth
                Status = "✓ SUCCESS"
                Color = "Green"
            }
            Write-Host "✓ Booth $($b.Booth) completed successfully" -ForegroundColor Green
        } else {
            $failCount++
            $results += @{
                Booth = $b.Booth
                Status = "✗ FAILED"
                Color = "Red"
            }
            Write-Host "✗ Booth $($b.Booth) failed" -ForegroundColor Red
        }
    } catch {
        $failCount++
        $results += @{
            Booth = $b.Booth
            Status = "✗ ERROR"
            Color = "Red"
        }
        Write-Host "✗ Error processing Booth $($b.Booth): $_" -ForegroundColor Red
    }
    
    Write-Host ("-" * 80) -ForegroundColor Gray
}

# Summary
Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "BATCH PROCESSING SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Total Booths: $($booths.Count)" -ForegroundColor White
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red

Write-Host "`nDetailed Results:" -ForegroundColor Cyan
foreach ($result in $results) {
    Write-Host "  Booth $($result.Booth): $($result.Status)" -ForegroundColor $result.Color
}

Write-Host "============================================================================" -ForegroundColor Cyan

if ($failCount -eq 0) {
    Write-Host "✓ All booths processed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠ Some booths failed to process. Check logs above for details." -ForegroundColor Yellow
    exit 1
}
