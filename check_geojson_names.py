import json

# Check GeoJSON ward names
print("=" * 80)
print("CHECKING GEOJSON WARD NAMES")
print("=" * 80)

try:
    with open('public/data/ward-boundaries.geojson') as f:
        geojson = json.load(f)
    
    print(f"\nTotal features in ward GeoJSON: {len(geojson.get('features', []))}")
    
    # Get unique ward names
    ward_names = set()
    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        
        # Try different possible field names
        ward_name = props.get('WardName') or props.get('Ward_Name') or props.get('name') or props.get('NAME')
        assembly = props.get('AC_Name') or props.get('AssemblyName') or props.get('Assembly')
        
        if ward_name:
            ward_names.add(ward_name)
            if assembly and 'R K' in assembly or 'RK' in assembly or 'Puram' in str(assembly):
                print(f"\nFeature properties for R K Puram area:")
                print(f"  Ward name field: {ward_name}")
                print(f"  Assembly field: {assembly}")
                print(f"  All properties: {list(props.keys())}")
    
    print(f"\n\nAll unique ward names in GeoJSON:")
    for name in sorted(ward_names):
        if 'puram' in name.lower() or 'munirka' in name.lower() or 'vasant' in name.lower():
            print(f"  - '{name}' ⭐")
        else:
            print(f"  - '{name}'")
            
except FileNotFoundError:
    print("Ward boundaries GeoJSON not found!")

# Now check electoral data ward names
print("\n" + "=" * 80)
print("CHECKING ELECTORAL DATA WARD NAMES (R K Puram assembly)")
print("=" * 80)

with open('public/data/electoral-data.json') as f:
    electoral_data = json.load(f)

rk_wards = set()
for booth in electoral_data:
    if booth.get('AssemblyName') == 'R K Puram' and booth.get('Ward Name'):
        rk_wards.add(booth['Ward Name'])

print("\nR K Puram assembly ward names in electoral data:")
for name in sorted(rk_wards):
    print(f"  - '{name}'")

print("\n" + "=" * 80)
print("POTENTIAL MISMATCH DETECTION")
print("=" * 80)
print("\nIf GeoJSON has 'R.K. PURAM' but electoral data has 'RK Puram',")
print("the map won't find the electoral data and will use default colors!")
