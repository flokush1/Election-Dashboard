# Quick Booth Integration Script
# Usage: .\process_booth.ps1 -BoothNumber 103 -Block E -Locality "B.K.Dutt Colony" -GeoJSONFile "Block-E, B.K.Dutt Colony.geojson"

param(
    [Parameter(Mandatory=$true)]
    [int]$BoothNumber,
    
    [Parameter(Mandatory=$false)]
    [string]$Assembly = "New Delhi",
    
    [Parameter(Mandatory=$true)]
    [string]$Locality,
    
    [Parameter(Mandatory=$false)]
    [string]$Block,
    
    [Parameter(Mandatory=$true)]
    [string]$GeoJSONFile,
    
    [Parameter(Mandatory=$false)]
    [string]$PredictionsCSV = "predictions_new_delhi.csv",
    
    [Parameter(Mandatory=$false)]
    [string]$PlotField = "plot_no."
)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Processing Booth $BoothNumber Integration" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan

# Build command
$cmd = "python integrate_booth_predictions.py --booth $BoothNumber --assembly `"$Assembly`" --locality `"$Locality`" --geojson `"$GeoJSONFile`" --predictions `"$PredictionsCSV`" --plot-field `"$PlotField`""

if ($Block) {
    $cmd += " --block `"$Block`""
}

Write-Host "`nExecuting: $cmd`n" -ForegroundColor Yellow

# Execute
Invoke-Expression $cmd

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n============================================================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS: Booth $BoothNumber integration completed!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    
    # Show output file location
    $outputDir = "public\data\geospatial"
    $assemblyShort = $Assembly -replace ' ', ''
    $blockPart = if ($Block) { "Block${Block}_" } else { "" }
    $outputFile = "$outputDir\${assemblyShort}_${blockPart}Booth_${BoothNumber}_Plots_With_Predictions.geojson"
    
    if (Test-Path $outputFile) {
        $fileSize = (Get-Item $outputFile).Length / 1KB
        Write-Host "`nOutput file: $outputFile" -ForegroundColor Cyan
        Write-Host "File size: $([math]::Round($fileSize, 2)) KB" -ForegroundColor Cyan
    }
} else {
    Write-Host "`n============================================================================" -ForegroundColor Red
    Write-Host "✗ ERROR: Booth $BoothNumber integration failed!" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
}

exit $exitCode
