# Booth-Level Voter Prediction Integration

This guide explains how to integrate voter predictions with GeoJSON plot/building data for any booth in the New Delhi Parliamentary constituency.

## Overview

The integration process matches voter predictions from CSV files to geographic plot boundaries in GeoJSON format, creating enriched geospatial data that can be visualized on interactive maps.

## Files Created

### Core Script
- **`integrate_booth_predictions.py`** - Generic Python script that works for any booth

### Output Files
Generated files are saved to `public/data/geospatial/` with naming convention:
```
{Assembly}_{Block}Booth_{BoothNumber}_Plots_With_Predictions.geojson
```

Example: `NewDelhi_BlockE_Booth_103_Plots_With_Predictions.geojson`

## Usage

### Basic Command Structure
```bash
python integrate_booth_predictions.py \
  --booth <BOOTH_NUMBER> \
  --assembly "<ASSEMBLY_NAME>" \
  --locality "<LOCALITY_NAME>" \
  --block "<BLOCK_ID>" \
  --geojson "<INPUT_GEOJSON_FILE>" \
  --predictions "<CSV_FILE>" \
  --plot-field "<PLOT_FIELD_NAME>"
```

### Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--booth` | ✅ Yes | Booth number (matches `partno` in CSV) | `103` |
| `--assembly` | ⚠️ Recommended | Assembly constituency name | `"New Delhi"` |
| `--locality` | ⚠️ Recommended | Locality/colony name | `"B.K.Dutt Colony"` |
| `--block` | ⚠️ Optional | Block identifier (A, B, C, E, etc.) | `"E"` |
| `--geojson` | ✅ Yes | Input GeoJSON file path | `"Block-E, B.K.Dutt Colony.geojson"` |
| `--predictions` | ⚠️ Optional | Predictions CSV file (default: `predictions_new_delhi.csv`) | `"predictions_new_delhi.csv"` |
| `--plot-field` | ⚠️ Optional | GeoJSON property for plot number (default: `plot_no.`) | `"plot_no."` or `"PLOT_NO"` |
| `--output` | ⚠️ Optional | Custom output path (auto-generated if not provided) | `"custom_output.geojson"` |

## Examples

### Example 1: Block-E, B.K. Dutt Colony (Booth 103)
```bash
python integrate_booth_predictions.py \
  --booth 103 \
  --assembly "New Delhi" \
  --locality "B.K.Dutt Colony" \
  --block "E" \
  --geojson "Block-E, B.K.Dutt Colony.geojson" \
  --plot-field "plot_no."
```

### Example 2: R.K. Puram Booth 17 (Already implemented)
```bash
python integrate_booth_predictions.py \
  --booth 17 \
  --assembly "R K Puram" \
  --locality "Shanti Niketan" \
  --geojson "Shanti Niketan Plot Data.geojson" \
  --predictions "predictions_r_k_puram.csv" \
  --plot-field "PLOT_NO"
```

### Example 3: Any future booth
```bash
python integrate_booth_predictions.py \
  --booth <YOUR_BOOTH_NUMBER> \
  --assembly "<ASSEMBLY_NAME>" \
  --locality "<LOCALITY_NAME>" \
  --block "<BLOCK_ID>" \
  --geojson "<YOUR_GEOJSON_FILE.geojson>"
```

## Input Data Requirements

### 1. GeoJSON File Format
Your GeoJSON should be a `FeatureCollection` with plot/building polygons containing:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "plot_no.": "172",
        "block_no": "E",
        "loacalityn": "B.K.Dutt Colony",
        "booth_no.": 103
      },
      "geometry": {
        "type": "MultiPolygon",
        "coordinates": [...]
      }
    }
  ]
}
```

**Key Property Fields:**
- `plot_no.` (or `PLOT_NO`) - Plot/building number for matching
- `block_no` - Block identifier (optional)
- `loacalityn` - Locality name (optional)
- `booth_no.` - Booth number (optional)

### 2. Predictions CSV Format
Your CSV should contain voter-level predictions with these columns:

**Required Columns:**
- `voters id` - Unique voter ID
- `partno` - Booth number (this is the booth identifier!)
- `house number` - Plot/house number (for matching to GeoJSON)
- `name` - Voter name
- `age`, `gender` - Demographics
- `turnout_prob` - Turnout probability
- `prob_BJP`, `prob_Congress`, `prob_AAP`, `prob_Others`, `prob_NOTA` - Party probabilities

**Optional Columns:**
- `assembly name`, `Locality`, `section no & road name` - For additional filtering
- `religion`, `caste`, `economic_category` - For enrichment

## Output GeoJSON Structure

The script enriches each plot with:

### Plot-Level Aggregates
```json
{
  "properties": {
    "voter_count": 7,
    "avg_turnout_prob": 0.5367,
    "avg_prob_BJP": 0.29,
    "avg_prob_Congress": 0.2071,
    "avg_prob_AAP": 0.4967,
    "avg_prob_Others": 0.0095,
    "avg_prob_NOTA": 0.0067,
    "predicted_winner": "AAP",
    "winner_probability": 0.4967,
    "voters": [...]
  }
}
```

### Individual Voter Data
```json
{
  "voters": [
    {
      "voter_id": "IJE0974675",
      "name": "KAMLA VOHRA",
      "age": 96,
      "gender": "FEMALE",
      "religion": "SIKH",
      "caste": "NO CASTE SYSTEM",
      "economic": "MIDDLE CLASS",
      "turnout_prob": 0.8188,
      "prob_BJP": 0.0592,
      "prob_Congress": 0.0819,
      "prob_AAP": 0.8515,
      "prob_Others": 0.0013,
      "prob_NOTA": 0.0062,
      "predicted_party": "AAP"
    }
  ]
}
```

## Matching Logic

The script uses intelligent matching to connect voters to plots:

1. **Normalization**: Removes spaces, hyphens, prefixes (HOUSE NO, H NO, BLOCK)
2. **Case-insensitive**: Converts to uppercase for matching
3. **Fallback**: Tries exact match if normalized match fails
4. **Reports**: Shows matched vs unmatched plots

### Plot Number Variations Handled
- `172` ↔ `172`
- `E-172` ↔ `172`
- `HOUSE NO E-172` ↔ `172`
- `E 172` ↔ `172`
- `-170/172` ↔ `170172`

## Statistics Output

The script provides:
- ✅ Total voters matched
- ✅ Plots with/without voter data
- ✅ Average turnout and party support probabilities
- ✅ Top 5 plots by BJP/Congress/AAP support
- ✅ Sample enriched feature preview

## Troubleshooting

### Issue: "No voters found matching the criteria"
**Solutions:**
1. Verify booth number exists in CSV: `partno` column
2. Check assembly name spelling (case-insensitive but must match)
3. Verify locality name in CSV `Locality` column

### Issue: "No voters were matched to plots"
**Solutions:**
1. Check plot field name: Use `--plot-field` parameter
2. Examine house number format in CSV vs GeoJSON
3. Add debug output to see normalized values

### Issue: Missing predictions CSV columns
**Solutions:**
1. Ensure CSV has all required columns listed above
2. Check column name spelling (case-sensitive)
3. Verify booth numbers are in `partno` column

## Adding New Booths - Quick Checklist

- [ ] Obtain GeoJSON file with plot boundaries
- [ ] Verify booth number in `partno` column of predictions CSV
- [ ] Identify assembly name and locality
- [ ] Check plot number field name in GeoJSON properties
- [ ] Run integration script with appropriate parameters
- [ ] Verify output file created in `public/data/geospatial/`
- [ ] Check statistics output for data quality

## Integration with Frontend

Once the enriched GeoJSON is created, you can:

1. **Add to data sources** in your React components
2. **Display on maps** using Leaflet/MapLibre
3. **Color-code plots** by predicted winner
4. **Show voter details** in popups/tooltips
5. **Filter by party support** thresholds
6. **Aggregate statistics** at booth level

Example component usage:
```javascript
import boothData from 'public/data/geospatial/NewDelhi_BlockE_Booth_103_Plots_With_Predictions.geojson';

// Color plots by predicted winner
const getColor = (feature) => {
  const winner = feature.properties.predicted_winner;
  return winner === 'BJP' ? '#FF9933' : 
         winner === 'AAP' ? '#0066CC' : 
         winner === 'Congress' ? '#19CD19' : '#CCCCCC';
};
```

## Batch Processing Multiple Booths

To process multiple booths, create a batch script:

**Windows (PowerShell):**
```powershell
# process_all_booths.ps1
$booths = @(
    @{Booth=103; Block="E"; Locality="B.K.Dutt Colony"; GeoJSON="Block-E, B.K.Dutt Colony.geojson"},
    @{Booth=104; Block="C"; Locality="B.K.Dutt Colony"; GeoJSON="Block-C, B.K.Dutt Colony.geojson"}
)

foreach ($b in $booths) {
    python integrate_booth_predictions.py `
      --booth $b.Booth `
      --assembly "New Delhi" `
      --locality $b.Locality `
      --block $b.Block `
      --geojson $b.GeoJSON
}
```

**Linux/Mac (Bash):**
```bash
#!/bin/bash
# process_all_booths.sh

declare -a booths=(
  "103:E:B.K.Dutt Colony:Block-E, B.K.Dutt Colony.geojson"
  "104:C:B.K.Dutt Colony:Block-C, B.K.Dutt Colony.geojson"
)

for booth_info in "${booths[@]}"; do
  IFS=':' read -r booth block locality geojson <<< "$booth_info"
  python integrate_booth_predictions.py \
    --booth "$booth" \
    --assembly "New Delhi" \
    --locality "$locality" \
    --block "$block" \
    --geojson "$geojson"
done
```

## Notes

- Script automatically creates output directory if it doesn't exist
- UTF-8 encoding preserved for Hindi/regional language names
- NaN values handled gracefully
- Memory efficient for large CSV files (streaming possible if needed)
- Plot-level aggregates use mean of all voters in that plot
