import json
import math

# Load the file with NaN values
print("Loading GeoJSON file...")
with open('public/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson', 'r') as f:
    content = f.read()

# Replace NaN with null (valid JSON)
print("Replacing NaN values with null...")
content = content.replace(': NaN,', ': null,')
content = content.replace(': NaN}', ': null}')

# Parse to verify it's valid JSON now
print("Validating JSON...")
try:
    data = json.loads(content)
    print(f"✓ Valid JSON with {len(data['features'])} features")
except json.JSONDecodeError as e:
    print(f"✗ Still invalid JSON: {e}")
    exit(1)

# Also clean up any NaN in the data structure itself
def clean_nan(obj):
    """Recursively replace NaN with None in nested structures"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

print("Cleaning NaN from data structure...")
cleaned_data = clean_nan(data)

# Save the cleaned version
print("Saving cleaned GeoJSON...")
with open('public/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson', 'w') as f:
    json.dump(cleaned_data, f, indent=2)

print("✓ Successfully cleaned GeoJSON file!")
print(f"  Total features: {len(cleaned_data['features'])}")

# Check a sample feature
sample = cleaned_data['features'][0]
print(f"\nSample feature properties:")
for key, value in sample['properties'].items():
    if value is None:
        print(f"  {key}: null (was NaN)")
