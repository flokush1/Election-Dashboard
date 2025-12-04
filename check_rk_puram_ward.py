import pandas as pd
import json

# Read the RK Puram data
df = pd.read_excel('VoterID_Data_Assembly/VoterID_Data_RKPuram.xlsx')

print("=" * 80)
print("RK PURAM ASSEMBLY - WARD ANALYSIS")
print("=" * 80)

if 'Ward Name' in df.columns and 'Winner' in df.columns:
    # Get unique wards
    wards = df['Ward Name'].unique()
    print(f"\nTotal wards in RK Puram Assembly: {len(wards)}")
    print(f"Ward Names: {sorted(wards)}\n")
    
    # Analyze each ward
    for ward in sorted(wards):
        ward_data = df[df['Ward Name'] == ward]
        print(f"\n{'=' * 80}")
        print(f"WARD: {ward}")
        print(f"{'=' * 80}")
        print(f"Total booths: {len(ward_data)}")
        
        # Winner distribution
        winner_counts = ward_data['Winner'].value_counts()
        print(f"\nBooths won by each party:")
        for party, count in winner_counts.items():
            percentage = (count / len(ward_data)) * 100
            print(f"  {party}: {count} booths ({percentage:.1f}%)")
        
        # Most booths won by
        leading_party = winner_counts.idxmax()
        print(f"\n✓ LEADING PARTY (Most booths won): {leading_party}")
        
        # Total votes
        if 'Total_Polled' in ward_data.columns:
            total_votes = ward_data['Total_Polled'].sum()
            print(f"\nTotal votes polled: {total_votes:,}")
            
            # Calculate party votes
            party_votes = {}
            for party in ['BJP', 'AAP', 'Congress', 'Others']:
                ratio_col = f'{party}_Ratio'
                if ratio_col in ward_data.columns:
                    party_votes[party] = (ward_data[ratio_col] * ward_data['Total_Polled']).sum()
            
            print(f"\nTotal votes by party:")
            for party, votes in sorted(party_votes.items(), key=lambda x: x[1], reverse=True):
                percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
                print(f"  {party}: {votes:,.0f} votes ({percentage:.1f}%)")
            
            top_party_by_votes = max(party_votes.items(), key=lambda x: x[1])[0]
            print(f"\n✓ TOP PARTY (Most votes): {top_party_by_votes}")
            
            if leading_party != top_party_by_votes:
                print(f"\n⚠️  MISMATCH DETECTED!")
                print(f"    - Leading by booths won: {leading_party}")
                print(f"    - Leading by total votes: {top_party_by_votes}")
                print(f"    - Map should show: {leading_party} (booths won)")

else:
    print("Required columns not found in data")
    print(f"Available columns: {df.columns.tolist()}")
