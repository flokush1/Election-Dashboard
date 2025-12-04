# Booth-wise Analysis Integration - README

## Changes Made

### Backend Changes (model_api.py)

Added a new endpoint `/api/booth-statistics/<assembly_name>/<booth_number>` that:
- Reads the actual CSV prediction file (e.g., `newdelhi_voter_predictions.csv`)
- Filters data by the `Booth_ID` (or `PartNo`) column matching the requested booth number
- Aggregates real voter-level predictions into booth-level statistics:
  - Total voters count
  - Average turnout probability
  - Average party probabilities (BJP, Congress, AAP, Others, NOTA)
  - Expected votes per party based on turnout
  - Predicted winner and margin
  - Demographics breakdown (age groups, gender, religion, caste, economic categories)

### Frontend Changes (BoothLevel.jsx)

Updated the booth-level dashboard to:
1. **Fetch real booth statistics** - Added `loadBoothStatistics()` function that calls the new API
2. **Display CSV data** - Modified the "Key Statistics" section to show:
   - Total Voters from CSV (with PartNo reference)
   - Expected Turnout with probability percentage
   - Predicted Winner from CSV predictions
   - Data Source indicator (CSV vs Historical)
3. **Update charts** - Modified all charts to prefer CSV data when available:
   - Party Vote Share chart now uses expected votes from predictions
   - Demographics charts (Age, Gender, Religion, Caste) use CSV aggregations
   - Green badges indicate when CSV data is being displayed

## How It Works

### Booth ID Mapping
- The CSV file uses `Booth_ID` column (values like "1", "2", "102", "103", etc.)
- When you select "Booth 103" in the UI, it fetches data where `Booth_ID == "103"`
- The PartNo from the CSV matches the booth number selected

### Example Data Flow
1. User selects "Booth 103" from dropdown
2. Frontend calls `/api/booth-statistics/New Delhi/103`
3. Backend loads `newdelhi_voter_predictions.csv`
4. Filters 785 voters where `Booth_ID == "103"`
5. Calculates aggregated statistics:
   - Average `prob_BJP`, `prob_Congress`, `prob_AAP`, etc.
   - Expected votes = (turnout probability × total voters) × party probability
   - Demographics counts from actual voter records
6. Returns JSON with all booth-level statistics
7. Frontend displays real data instead of random/historical values

## Testing

To test the integration:

1. **Restart the Flask server** to load the new endpoint:
   ```powershell
   # Stop the current server (Ctrl+C in the python terminal)
   # Then restart:
   python model_api.py
   ```

2. **Test the API endpoint directly**:
   ```powershell
   curl "http://127.0.0.1:5000/api/booth-statistics/New%20Delhi/103"
   ```

3. **Open the dashboard** and navigate to:
   - Parliament Level → New Delhi
   - Assembly Level → Select any assembly
   - Ward Level → Select a ward
   - Booth Level → Select Booth 102, 103, or any booth number

4. **Verify**:
   - Look for green "CSV" badges on chart headers
   - Check "Data Source" card shows "CSV Predictions"
   - Party probabilities should show percentages from actual predictions
   - Total Voters should match the CSV row count for that booth

## CSV File Structure

The `newdelhi_voter_predictions.csv` file contains:
- **123 unique booth numbers** (1-123)
- **Booth_ID column** - matches booth/part number
- **Prediction columns**: `prob_BJP`, `prob_Congress`, `prob_AAP`, `prob_Others`, `prob_NOTA`, `turnout_prob`
- **Demographics**: `Age`, `gender`, `Religion`, `Caste`, `Economic`, `Locality`
- **Voter info**: `Voter_ID`, `name`, `relation_name`, `house_number`, etc.

Example booth counts:
- Booth 102: 1,123 voters
- Booth 103: 785 voters

## Notes

- The system gracefully falls back to electoral-data.json if CSV data is unavailable
- All calculations are done server-side for performance
- Demographics are aggregated from actual voter records, not synthetic data
- The predicted winner is determined by the party with highest average probability
