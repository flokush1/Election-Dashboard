import pandas as pd

df = pd.read_excel('NewDelhi_Parliamentary_Data.xlsx')

wards = sorted(set(df['Ward Name'].dropna().unique()))
print('Total wards:', len(wards))
print('\nWards containing Park or Chitt:')
for w in wards:
    lw = str(w).lower()
    if 'park' in lw or 'chitt' in lw or 'c r park' in lw or 'c.r.' in lw or 'cr park' in lw or 'c r p' in lw:
        print(' -', w)

print('\nAssemblies containing these wards:')
mask = df['Ward Name'].astype(str).str.contains('Chitt|C R Park|C.R.|CR Park|Park', case=False, na=False)
for aname, g in df[mask].groupby('AssemblyName'):
    print(' *', aname, '-> wards:', sorted(set(g['Ward Name'])))

candidates = [
  'Chittaranjan Park', 'Chittaranjan Pk', 'C R Park', 'CR Park', 'C.R. Park', 'Chitaranjan Park', 'Chittranjan Park'
]
for cand in candidates:
    sub = df[df['Ward Name'].astype(str).str.lower() == cand.lower()]
    if not sub.empty:
        print(f"\nWard '{cand}' found: booths {len(sub)}")
        print(sub['Winner'].value_counts())
