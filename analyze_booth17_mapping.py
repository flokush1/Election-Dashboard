import pandas as pd
import json
import re

# Load voter data
print("Loading voter data...")
df = pd.read_excel('VoterID_Data_Assembly/VoterID_Data_RKPuram.xlsx')
booth17 = df[df['partno'] == 17].copy()
print(f"Total voters in Booth 17: {len(booth17)}\n")

# Extract plot numbers from addresses
def extract_plot_info(addr):
    """Extract plot number and street from address"""
    addr_str = str(addr)
    
    # Pattern: X/Y format (e.g., 1/10, 3/15)
    match = re.search(r'(\d+)/(\d+)', addr_str)
    if match:
        street = match.group(1)
        plot = match.group(2)
        return f"{street}/{plot}", street, plot
    return None, None, None

booth17['plot_address'], booth17['street'], booth17['plot_num'] = zip(*booth17['full address'].apply(extract_plot_info))

# Group by plot
plot_groups = booth17.groupby('plot_address').agg({
    'voters id': 'count',
    'name': list,
    'age': list,
    'gender': list,
    'full address': 'first',
    'street': 'first',
    'plot_num': 'first'
}).rename(columns={'voters id': 'voter_count'})

plot_groups = plot_groups.sort_values('voter_count', ascending=False)

print("Top 30 plots by voter count:")
print("="*80)
for idx, row in plot_groups.head(30).iterrows():
    if idx:
        print(f"{idx:15s} | Street {row['street']:2s} Plot {row['plot_num']:3s} | {row['voter_count']:3d} voters")

# Load GeoJSON data
print("\n\nLoading GeoJSON plot data...")
with open('public/data/geospatial/Shanti_Niketan_Plots.geojson', 'r') as f:
    geojson_data = json.load(f)

# Create mapping of GeoJSON plots
geojson_plots = {}
for feature in geojson_data['features']:
    props = feature['properties']
    road_no = props.get('Road_No', '')
    plot_no = props.get('PLOT_NO', '')
    
    if road_no and plot_no:
        key = f"{road_no}/{plot_no}"
        geojson_plots[key] = {
            'NAME': props.get('NAME', ''),
            'AREA_SQMTR': props.get('AREA_SQMTR', ''),
            'Parcel_No': props.get('Parcel_No', ''),
            'geometry': feature['geometry']
        }

print(f"\nTotal plots in GeoJSON: {len(geojson_plots)}")
print(f"Total voter address plots: {len(plot_groups)}")

# Find matches
print("\n\nMatching voter plots with GeoJSON plots:")
print("="*80)
matched = 0
unmatched_voter = 0
unmatched_geo = 0

matched_plots = []
for plot_addr in plot_groups.index:
    if plot_addr and plot_addr in geojson_plots:
        matched += 1
        voter_info = plot_groups.loc[plot_addr]
        geo_info = geojson_plots[plot_addr]
        matched_plots.append({
            'plot_address': plot_addr,
            'voter_count': voter_info['voter_count'],
            'building_type': geo_info['NAME'],
            'area_sqm': geo_info['AREA_SQMTR']
        })
    elif plot_addr:
        unmatched_voter += 1

unmatched_geo = len(geojson_plots) - matched

print(f"✓ Matched plots: {matched}")
print(f"✗ Voter plots not in GeoJSON: {unmatched_voter}")
print(f"✗ GeoJSON plots without voters: {unmatched_geo}")

print("\n\nTop 20 matched plots with voter counts:")
print("="*80)
matched_df = pd.DataFrame(matched_plots).sort_values('voter_count', ascending=False)
for idx, row in matched_df.head(20).iterrows():
    print(f"{row['plot_address']:15s} | {row['voter_count']:3d} voters | {row['building_type']:20s} | {float(row['area_sqm']):8.1f} sqm")

# Create enhanced GeoJSON with voter data
print("\n\nCreating enhanced GeoJSON with voter data...")
enhanced_features = []

for feature in geojson_data['features']:
    props = feature['properties']
    road_no = props.get('Road_No', '')
    plot_no = props.get('PLOT_NO', '')
    key = f"{road_no}/{plot_no}"
    
    # Add voter data if available
    if key in plot_groups.index:
        voter_info = plot_groups.loc[key]
        props['voter_count'] = int(voter_info['voter_count'])
        props['voters'] = [
            {
                'name': name,
                'age': int(age) if pd.notna(age) else None,
                'gender': gender
            }
            for name, age, gender in zip(voter_info['name'], voter_info['age'], voter_info['gender'])
            if pd.notna(name)
        ]
    else:
        props['voter_count'] = 0
        props['voters'] = []
    
    enhanced_features.append(feature)

enhanced_geojson = {
    "type": "FeatureCollection",
    "name": "Shanti Niketan Plot Data with Voters",
    "crs": geojson_data['crs'],
    "features": enhanced_features
}

# Save enhanced GeoJSON
output_path = 'public/data/geospatial/RKPuram_Booth_17_Plots_With_Voters.geojson'
with open(output_path, 'w') as f:
    json.dump(enhanced_geojson, f, indent=2)

print(f"✓ Enhanced GeoJSON saved to: {output_path}")
print(f"  Total features: {len(enhanced_features)}")
print(f"  Features with voters: {sum(1 for f in enhanced_features if f['properties']['voter_count'] > 0)}")
print(f"  Total voters included: {sum(f['properties']['voter_count'] for f in enhanced_features)}")

print("\n\nSample enhanced feature:")
sample = next((f for f in enhanced_features if f['properties']['voter_count'] > 0), None)
if sample:
    print(json.dumps({
        'properties': {
            k: v if k != 'voters' else f"{len(v)} voters (showing first 3)" 
            for k, v in sample['properties'].items()
        },
        'sample_voters': sample['properties']['voters'][:3]
    }, indent=2))
