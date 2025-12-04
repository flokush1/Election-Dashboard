import pandas as pd
import json
import re

print("="*80)
print("RK PURAM BOOTH 17 - VOTER DATA + AI PREDICTIONS INTEGRATION")
print("="*80)

# Load predictions
print("\n1. Loading voter predictions...")
predictions_df = pd.read_csv('rk_puram_voter_predictions.csv')
booth17_pred = predictions_df[predictions_df['Booth_ID'] == 17].copy()
print(f"   ✓ Loaded {len(booth17_pred)} voter predictions for Booth 17")

# Key columns for predictions
pred_cols = ['Voter_ID', 'name', 'Age', 'gender', 'full address', 
             'turnout_prob', 'prob_BJP', 'prob_Congress', 'prob_AAP', 
             'prob_Others', 'prob_NOTA', 'Religion', 'Caste', 'Economic', 'Income']

# Extract plot info
def extract_plot_info(addr):
    addr_str = str(addr)
    match = re.search(r'(\d+)/(\d+)', addr_str)
    if match:
        street = match.group(1)
        plot = match.group(2)
        return f"{street}/{plot}", street, plot
    return None, None, None

booth17_pred['plot_address'], booth17_pred['street'], booth17_pred['plot_num'] = \
    zip(*booth17_pred['full address'].apply(extract_plot_info))

# Load existing GeoJSON
print("\n2. Loading GeoJSON plot data...")
with open('public/data/geospatial/Shanti_Niketan_Plots.geojson', 'r') as f:
    geojson_data = json.load(f)
print(f"   ✓ Loaded {len(geojson_data['features'])} plot features")

# Create enhanced features with predictions
print("\n3. Enriching plots with voter predictions...")
enhanced_features = []
total_voters_with_predictions = 0
plots_with_data = 0

for feature in geojson_data['features']:
    props = feature['properties']
    road_no = props.get('Road_No', '')
    plot_no = props.get('PLOT_NO', '')
    key = f"{road_no}/{plot_no}"
    
    # Get voters for this plot
    plot_voters = booth17_pred[booth17_pred['plot_address'] == key]
    
    if len(plot_voters) > 0:
        plots_with_data += 1
        total_voters_with_predictions += len(plot_voters)
        
        # Calculate aggregate statistics
        avg_turnout = plot_voters['turnout_prob'].mean()
        avg_bjp = plot_voters['prob_BJP'].mean()
        avg_congress = plot_voters['prob_Congress'].mean()
        avg_aap = plot_voters['prob_AAP'].mean()
        avg_others = plot_voters['prob_Others'].mean()
        avg_nota = plot_voters['prob_NOTA'].mean()
        
        # Determine predicted winner
        party_probs = {
            'BJP': avg_bjp,
            'Congress': avg_congress,
            'AAP': avg_aap,
            'Others': avg_others,
            'NOTA': avg_nota
        }
        predicted_winner = max(party_probs, key=party_probs.get)
        
        # Add enriched properties
        props['voter_count'] = len(plot_voters)
        props['avg_turnout_prob'] = round(avg_turnout, 4)
        props['avg_prob_BJP'] = round(avg_bjp, 4)
        props['avg_prob_Congress'] = round(avg_congress, 4)
        props['avg_prob_AAP'] = round(avg_aap, 4)
        props['avg_prob_Others'] = round(avg_others, 4)
        props['avg_prob_NOTA'] = round(avg_nota, 4)
        props['predicted_winner'] = predicted_winner
        props['winner_probability'] = round(party_probs[predicted_winner], 4)
        
        # Add individual voters with predictions
        props['voters'] = []
        for _, voter in plot_voters.iterrows():
            voter_data = {
                'voter_id': voter['Voter_ID'],
                'name': voter['name'],
                'age': int(voter['Age']) if pd.notna(voter['Age']) else None,
                'gender': voter['gender'],
                'religion': voter['Religion'] if pd.notna(voter['Religion']) else None,
                'caste': voter['Caste'] if pd.notna(voter['Caste']) else None,
                'economic': voter['Economic'] if pd.notna(voter['Economic']) else None,
                'income': voter['Income'] if pd.notna(voter['Income']) else None,
                'turnout_prob': round(float(voter['turnout_prob']), 4),
                'prob_BJP': round(float(voter['prob_BJP']), 4),
                'prob_Congress': round(float(voter['prob_Congress']), 4),
                'prob_AAP': round(float(voter['prob_AAP']), 4),
                'prob_Others': round(float(voter['prob_Others']), 4),
                'prob_NOTA': round(float(voter['prob_NOTA']), 4),
                'predicted_party': max(
                    [('BJP', voter['prob_BJP']), 
                     ('Congress', voter['prob_Congress']), 
                     ('AAP', voter['prob_AAP']),
                     ('Others', voter['prob_Others']),
                     ('NOTA', voter['prob_NOTA'])],
                    key=lambda x: x[1]
                )[0]
            }
            props['voters'].append(voter_data)
    else:
        # No voters for this plot
        props['voter_count'] = 0
        props['voters'] = []
    
    enhanced_features.append(feature)

# Save enhanced GeoJSON
enhanced_geojson = {
    "type": "FeatureCollection",
    "name": "RK Puram Booth 17 - Plots with Voter Predictions",
    "crs": geojson_data['crs'],
    "features": enhanced_features
}

output_path = 'public/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson'
with open(output_path, 'w') as f:
    json.dump(enhanced_geojson, f, indent=2)

print(f"   ✓ Enhanced {plots_with_data} plots with prediction data")
print(f"   ✓ Total voters with predictions: {total_voters_with_predictions}")

# Statistics
print("\n4. Booth 17 Aggregate Statistics:")
print("   " + "-"*60)
print(f"   Average Turnout Probability: {booth17_pred['turnout_prob'].mean():.2%}")
print(f"   Average BJP Support:         {booth17_pred['prob_BJP'].mean():.2%}")
print(f"   Average Congress Support:    {booth17_pred['prob_Congress'].mean():.2%}")
print(f"   Average AAP Support:         {booth17_pred['prob_AAP'].mean():.2%}")
print(f"   Average Others Support:      {booth17_pred['prob_Others'].mean():.2%}")
print(f"   Average NOTA:                {booth17_pred['prob_NOTA'].mean():.2%}")

# Top plots by party support
print("\n5. Top 10 Plots by Party Support:")
print("   " + "-"*60)

plot_stats = booth17_pred.groupby('plot_address').agg({
    'Voter_ID': 'count',
    'prob_BJP': 'mean',
    'prob_Congress': 'mean',
    'prob_AAP': 'mean'
}).rename(columns={'Voter_ID': 'voters'})

print("\n   BJP Strong Plots:")
for idx, (plot, row) in enumerate(plot_stats.nlargest(5, 'prob_BJP').iterrows(), 1):
    if plot:
        print(f"   {idx}. {plot:15s} - {row['voters']:3.0f} voters - {row['prob_BJP']:.1%} BJP support")

print("\n   Congress Strong Plots:")
for idx, (plot, row) in enumerate(plot_stats.nlargest(5, 'prob_Congress').iterrows(), 1):
    if plot:
        print(f"   {idx}. {plot:15s} - {row['voters']:3.0f} voters - {row['prob_Congress']:.1%} Congress support")

print("\n   AAP Strong Plots:")
for idx, (plot, row) in enumerate(plot_stats.nlargest(5, 'prob_AAP').iterrows(), 1):
    if plot:
        print(f"   {idx}. {plot:15s} - {row['voters']:3.0f} voters - {row['prob_AAP']:.1%} AAP support")

# Save to file
print(f"\n6. Output saved to: {output_path}")
print("\n" + "="*80)
print("✓ INTEGRATION COMPLETE!")
print("="*80)

# Sample enriched feature
print("\nSample Enhanced Feature:")
sample = next((f for f in enhanced_features if f['properties']['voter_count'] > 0), None)
if sample:
    print(json.dumps({
        'plot_address': f"{sample['properties']['Road_No']}/{sample['properties']['PLOT_NO']}",
        'building_type': sample['properties']['NAME'],
        'voter_count': sample['properties']['voter_count'],
        'predicted_winner': sample['properties']['predicted_winner'],
        'winner_probability': sample['properties']['winner_probability'],
        'avg_turnout': sample['properties']['avg_turnout_prob'],
        'party_support': {
            'BJP': sample['properties']['avg_prob_BJP'],
            'Congress': sample['properties']['avg_prob_Congress'],
            'AAP': sample['properties']['avg_prob_AAP']
        },
        'sample_voter': sample['properties']['voters'][0] if sample['properties']['voters'] else None
    }, indent=2))
