import pandas as pd

df = pd.read_csv('newdelhi_voter_predictions.csv')

print('Booth 102 voters:', len(df[df['Booth_ID'].astype(str) == '102']))
print('Booth 103 voters:', len(df[df['Booth_ID'].astype(str) == '103']))

print('\nSample from Booth 103:')
b103 = df[df['Booth_ID'].astype(str) == '103']
if len(b103) > 0:
    print(b103[['Voter_ID', 'name', 'Age', 'gender', 'Religion', 'Caste', 'prob_BJP', 'prob_AAP', 'prob_Congress']].head(3))
else:
    print('Booth 103 not found')
