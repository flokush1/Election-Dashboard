import pandas as pd
import json

# Load predictions
print("Loading RK Puram voter predictions...")
df = pd.read_csv('rk_puram_voter_predictions.csv')
print(f"Total predictions: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head(5))

# Find booth column
booth_col = None
for col in ['Part No', 'Booth', 'PartNo', 'part_no', 'booth_no']:
    if col in df.columns:
        booth_col = col
        print(f"\nFound booth column: {booth_col}")
        break

if booth_col:
    print(f"\nUnique booths: {sorted(df[booth_col].unique())}")
    
    # Filter for Booth 17
    booth17 = df[df[booth_col] == 17].copy()
    print(f"\nTotal Booth 17 predictions: {len(booth17)}")
    
    if len(booth17) > 0:
        print("\nSample Booth 17 predictions:")
        print(booth17.head(10))
        
        # Check for prediction columns
        pred_cols = [col for col in df.columns if 'pred' in col.lower() or 'aap' in col.lower() or 'bjp' in col.lower() or 'congress' in col.lower()]
        print(f"\nPrediction columns found: {pred_cols}")
        
        if pred_cols:
            print("\nPrediction distribution for Booth 17:")
            for col in pred_cols:
                if col in booth17.columns:
                    print(f"\n{col}:")
                    print(booth17[col].value_counts())
        
        # Try to match with voter data
        print("\n\nAttempting to match with voter IDs...")
        if 'Voter ID' in booth17.columns or 'voters id' in booth17.columns:
            voter_id_col = 'Voter ID' if 'Voter ID' in booth17.columns else 'voters id'
            print(f"Found voter ID column: {voter_id_col}")
            print(f"Sample voter IDs: {booth17[voter_id_col].head(10).tolist()}")
        else:
            print("No voter ID column found. Available columns:", booth17.columns.tolist())
else:
    print("\nCould not find booth column!")
    print("Available columns:", df.columns.tolist())
