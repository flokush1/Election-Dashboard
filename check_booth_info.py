import pandas as pd
import json

# Check prediction data
df = pd.read_csv('Prediction Models/RK Puram/enhanced_all_booth_predictions.csv')
booth17 = df[df['Booth_ID'] == 17]

print("Columns in prediction data:")
print(df.columns.tolist())

print("\n\nBooth 17 sample data:")
if len(booth17) > 0:
    print(booth17.head(5).to_string())
