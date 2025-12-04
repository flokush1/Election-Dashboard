import json

# Load the electoral data JSON
with open('public/data/electoral-data.json') as f:
    data = json.load(f)

# Simulate what the dataProcessor does
wards = {}
for booth in data:
    ward_name = booth.get('Ward Name')
    assembly = booth.get('AssemblyName')
    
    if assembly == 'R K Puram' and ward_name:
        if ward_name not in wards:
            wards[ward_name] = {
                'booths': [],
                'boothsWon': {'BJP': 0, 'AAP': 0, 'Congress': 0, 'Others': 0, 'Tie': 0},
                'partyVotes': {'BJP': 0, 'AAP': 0, 'Congress': 0, 'Others': 0, 'NOTA': 0},
                'totalVotes': 0
            }
        
        wards[ward_name]['booths'].append(booth)
        
        # Count booths won
        winner = booth.get('Winner')
        if winner in wards[ward_name]['boothsWon']:
            wards[ward_name]['boothsWon'][winner] += 1
        
        # Aggregate votes
        total_polled = booth.get('Total_Polled', 0)
        wards[ward_name]['totalVotes'] += total_polled
        for party in ['BJP', 'AAP', 'Congress', 'Others', 'NOTA']:
            ratio = booth.get(f'{party}_Ratio', 0)
            wards[ward_name]['partyVotes'][party] += ratio * total_polled

print("=" * 80)
print("R K PURAM ASSEMBLY - WARD DATA ANALYSIS")
print("=" * 80)

for ward_name, ward_data in sorted(wards.items()):
    print(f"\n{'='*80}")
    print(f"WARD: {ward_name}")
    print(f"{'='*80}")
    print(f"Total booths: {len(ward_data['booths'])}")
    
    print(f"\nBooths Won:")
    for party, count in sorted(ward_data['boothsWon'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {party}: {count}")
    
    # Winner by booths
    booths_won_sorted = [(p, c) for p, c in ward_data['boothsWon'].items() if p != 'Tie' and c > 0]
    if booths_won_sorted:
        booths_won_sorted.sort(key=lambda x: x[1], reverse=True)
        winner_by_booths = booths_won_sorted[0][0]
        print(f"\n✓ Winner by BOOTHS WON: {winner_by_booths}")
    
    print(f"\nParty Votes:")
    for party, votes in sorted(ward_data['partyVotes'].items(), key=lambda x: x[1], reverse=True):
        pct = (votes / ward_data['totalVotes'] * 100) if ward_data['totalVotes'] > 0 else 0
        print(f"  {party}: {votes:.0f} ({pct:.1f}%)")
    
    # Winner by votes
    votes_sorted = sorted(ward_data['partyVotes'].items(), key=lambda x: x[1], reverse=True)
    winner_by_votes = votes_sorted[0][0]
    print(f"\n✗ Winner by TOTAL VOTES: {winner_by_votes}")
    
    if winner_by_booths != winner_by_votes:
        print(f"\n⚠️  MISMATCH!")
        print(f"   Map should show: {winner_by_booths} (by booths won)")
        print(f"   If showing: {winner_by_votes} then the fix didn't work!")

print("\n" + "=" * 80)
print("Checking what the frontend will receive:")
print("=" * 80)

# Check what ward names are in the data
print(f"\nWard names in R K Puram assembly:")
for ward_name in sorted(wards.keys()):
    print(f"  - '{ward_name}'")
