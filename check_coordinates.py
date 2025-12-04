import json

# Check boundary structure
with open('public/data/geospatial/RKPuram_Booth_17_Boundary.geojson') as f:
    boundary = json.load(f)

print("=== BOUNDARY ===")
print(f"Type: {boundary['features'][0]['geometry']['type']}")
coords = boundary['features'][0]['geometry']['coordinates']
print(f"Structure: {len(coords)} polygons")
print(f"First polygon: {len(coords[0])} rings")  
print(f"First ring: {len(coords[0][0])} points")
print(f"Coordinate range (lon): {min(c[0] for c in coords[0][0]):.6f} to {max(c[0] for c in coords[0][0]):.6f}")
print(f"Coordinate range (lat): {min(c[1] for c in coords[0][0]):.6f} to {max(c[1] for c in coords[0][0]):.6f}")

# Check plots structure
with open('public/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson') as f:
    plots = json.load(f)

print(f"\n=== PLOTS ===")
print(f"Total features: {len(plots['features'])}")

first = plots['features'][0]
print(f"First feature type: {first['geometry']['type']}")
print(f"First feature plot: {first['properties'].get('Road_No')}/{first['properties'].get('PLOT_NO')}")
pcoords = first['geometry']['coordinates'][0]
print(f"First plot coords: {len(pcoords)} points")
print(f"First coord: {pcoords[0]}")

# Get all plot coordinate ranges
all_lons = []
all_lats = []
for feat in plots['features']:
    for point in feat['geometry']['coordinates'][0]:
        # Handle 3D coordinates (lon, lat, elevation)
        all_lons.append(point[0])
        all_lats.append(point[1])

print(f"All plots lon range: {min(all_lons):.6f} to {max(all_lons):.6f}")
print(f"All plots lat range: {min(all_lats):.6f} to {max(all_lats):.6f}")

# Check overlap
boundary_lon_min = min(c[0] for c in coords[0][0])
boundary_lon_max = max(c[0] for c in coords[0][0])
boundary_lat_min = min(c[1] for c in coords[0][0])
boundary_lat_max = max(c[1] for c in coords[0][0])

plots_lon_min = min(all_lons)
plots_lon_max = max(all_lons)
plots_lat_min = min(all_lats)
plots_lat_max = max(all_lats)

print(f"\n=== OVERLAP CHECK ===")
lon_overlap = not (plots_lon_max < boundary_lon_min or plots_lon_min > boundary_lon_max)
lat_overlap = not (plots_lat_max < boundary_lat_min or plots_lat_min > boundary_lat_max)

print(f"Longitude overlap: {lon_overlap}")
print(f"Latitude overlap: {lat_overlap}")
print(f"Geographic overlap: {lon_overlap and lat_overlap}")

if not (lon_overlap and lat_overlap):
    print("\n⚠️ WARNING: Boundary and plots don't overlap geographically!")
    print("This explains why clipping returns 0 features.")
