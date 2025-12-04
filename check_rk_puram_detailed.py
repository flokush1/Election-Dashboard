import json

# Load electoral data
with open('public/data/electoral-data.json') as f:
    data = json.load(f)

# Filter for RK Puram assembly wards (note: it's "R K Puram" with spaces)
rk_wards = {}
for booth in data:
    if booth.get('AssemblyName') == 'R K Puram' and booth.get('Ward Name'):
        ward_name = booth['Ward Name']
        if ward_name not in rk_wards:
            rk_wards[ward_name] = []
        rk_wards[ward_name].append(booth)

print("=" * 80)
print("RK PURAM ASSEMBLY - WARD BOOTH WINNERS ANALYSIS")
print("=" * 80)

for ward_name in sorted(rk_wards.keys()):
    booths = rk_wards[ward_name]
    print(f"\n{ward_name}: {len(booths)} booths")
    
    # Count winners
    winner_counts = {}
    for booth in booths:
        winner = booth.get('Winner', 'Unknown')
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
    
    print(f"  Booth Winners:")
    for party in sorted(winner_counts.keys(), key=lambda x: winner_counts[x], reverse=True):
        print(f"    {party}: {winner_counts[party]} booths")
    
    # Find leading party by booths won
    leading_party = max(winner_counts.items(), key=lambda x: x[1])[0]
    print(f"  ✓ LEADING PARTY (by booths won): {leading_party}")
    
    # Calculate total votes
    total_votes_by_party = {'BJP': 0, 'AAP': 0, 'Congress': 0, 'Others': 0}
    total_polled = 0
    
    for booth in booths:
        polled = booth.get('Total_Polled', 0)
        total_polled += polled
        for party in total_votes_by_party.keys():
            ratio = booth.get(f'{party}_Ratio', 0)
            total_votes_by_party[party] += ratio * polled
    
    print(f"  Total votes by party:")
    for party in sorted(total_votes_by_party.keys(), key=lambda x: total_votes_by_party[x], reverse=True):
        pct = (total_votes_by_party[party] / total_polled * 100) if total_polled > 0 else 0
        print(f"    {party}: {total_votes_by_party[party]:.0f} ({pct:.1f}%)")
    
    top_by_votes = max(total_votes_by_party.items(), key=lambda x: x[1])[0]
    
    if leading_party != top_by_votes:
        print(f"  ⚠️  MISMATCH:")
        print(f"      Leading by booths won: {leading_party}")
        print(f"      Leading by total votes: {top_by_votes}")
        print(f"      Map SHOULD show: {leading_party} (booths won)")
