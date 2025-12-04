import pandas as pd

df = pd.read_csv('Prediction Models/RK Puram/rk_puram_voter_predictions.csv')
print('Columns:', df.columns.tolist())
print(f'\nTotal rows: {len(df)}')
print('\nFirst 3 rows:')
print(df.head(3).to_string())

if 'Booth_ID' in df.columns:
    print(f'\nBooth 17 rows: {len(df[df["Booth_ID"] == 17])}')
