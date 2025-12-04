# Booth Configuration Template
# Copy this template and fill in details for each new booth you want to process

# ============================================================================
# BOOTH CONFIGURATION
# ============================================================================

# Basic Information
$boothNumber = 103                              # Booth number from 'partno' column in CSV
$assemblyName = "New Delhi"                     # Assembly constituency name
$locality = "B.K.Dutt Colony"                   # Locality/colony name
$block = "E"                                    # Block identifier (A, B, C, E, etc.)

# File Paths
$geojsonFile = "Block-E, B.K.Dutt Colony.geojson"   # Input GeoJSON file
$predictionsCSV = "predictions_new_delhi.csv"        # Predictions CSV file
$plotFieldName = "plot_no."                          # GeoJSON property for plot number

# ============================================================================
# EXECUTE INTEGRATION
# ============================================================================

Write-Host "Processing Booth $boothNumber..." -ForegroundColor Cyan

$cmd = "python integrate_booth_predictions.py " +
       "--booth $boothNumber " +
       "--assembly `"$assemblyName`" " +
       "--locality `"$locality`" " +
       "--geojson `"$geojsonFile`" " +
       "--predictions `"$predictionsCSV`" " +
       "--plot-field `"$plotFieldName`""

if ($block) {
    $cmd += " --block `"$block`""
}

Write-Host "Command: $cmd`n" -ForegroundColor Yellow
Invoke-Expression $cmd

# ============================================================================
# EXPECTED OUTPUT
# ============================================================================

# Output file will be created at:
# public/data/geospatial/NewDelhi_BlockE_Booth_103_Plots_With_Predictions.geojson

# ============================================================================
# USAGE NOTES
# ============================================================================

<#
1. Copy this template for each booth
2. Update the configuration variables above
3. Run the script: .\booth_config_template.ps1
4. Check the output in public/data/geospatial/

EXAMPLE CONFIGURATIONS:

# Booth 104, Block C, B.K. Dutt Colony
$boothNumber = 104
$block = "C"
$locality = "B.K.Dutt Colony"
$geojsonFile = "Block-C, B.K.Dutt Colony.geojson"

# Booth 25, Different Assembly
$boothNumber = 25
$assemblyName = "Jangpura"
$locality = "Jangpura Extension"
$block = "A"
$geojsonFile = "Jangpura-Block-A.geojson"

# Booth without block identifier
$boothNumber = 50
$assemblyName = "New Delhi"
$locality = "Pandara Road"
$block = ""
$geojsonFile = "Pandara-Road-Plots.geojson"
#>
