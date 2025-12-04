import pandas as pd

# Load the main parliamentary data
df = pd.read_excel('NewDelhi_Parliamentary_Data.xlsx')

print("=" * 80)
print("NEWDELHI PARLIAMENTARY DATA - COLUMNS")
print("=" * 80)
print(df.columns.tolist())

print("\n" + "=" * 80)
print("SAMPLE DATA (First 3 rows)")
print("=" * 80)
print(df.head(3))

# Check for RK Puram
print("\n" + "=" * 80)
print("RK PURAM ASSEMBLY - WARD ANALYSIS")
print("=" * 80)

rk_puram_data = df[df['AssemblyName'] == 'R K Puram']
print(f"\nTotal RK Puram booths: {len(rk_puram_data)}")

if len(rk_puram_data) > 0 and 'Ward Name' in df.columns:
    wards = rk_puram_data['Ward Name'].unique()
    print(f"Wards in RK Puram: {sorted([w for w in wards if pd.notna(w)])}")
    
    for ward_name in sorted([w for w in wards if pd.notna(w)]):
        ward_data = rk_puram_data[rk_puram_data['Ward Name'] == ward_name]
        print(f"\n{'=' * 80}")
        print(f"WARD: {ward_name}")
        print(f"{'=' * 80}")
        print(f"Total booths: {len(ward_data)}")
        
        # Winner distribution
        if 'Winner' in ward_data.columns:
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
                    print(f"    - Map should show: {leading_party} (correct)")
                    print(f"    - If map shows: {top_party_by_votes} (WRONG - this is the bug!)")
