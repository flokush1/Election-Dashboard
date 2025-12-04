import pandas as pd
import json
import shutil
import os

def convert_excel_to_json():
    """Convert Excel data to JSON format with proper data processing"""
    try:
        # Read the Excel file
        excel_path = 'NewDelhi_Parliamentary_Data.xlsx'
        if not os.path.exists(excel_path):
            print(f"Excel file not found at {excel_path}")
            return False
            
        df = pd.read_excel(excel_path)
        print(f"Loaded {len(df)} records from Excel")
        
        # Convert to JSON-serializable format
        data = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                value = row[col]
                # Handle NaN values
                if pd.isna(value):
                    record[col] = None
                # Handle numpy types
                elif hasattr(value, 'item'):
                    record[col] = value.item()
                else:
                    record[col] = value
            data.append(record)
        
        # Save to JSON
        os.makedirs('public/data', exist_ok=True)
        with open('public/data/electoral-data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted {len(data)} booth records to JSON")
        return True
        
    except Exception as e:
        print(f"Error converting Excel data: {e}")
        return False

def copy_geojson_files():
    """Copy GeoJSON files to public directory"""
    geojson_files = [
        ('../New Delhi_ACS_Boundaries.geojson', 'public/data/assembly-boundaries.geojson'),
        ('../New Delhi_PCS_Boundaries.geojson', 'public/data/parliament-boundaries.geojson'), 
        ('../New Delhi_Wards_Boundaries.geojson', 'public/data/ward-boundaries.geojson')
    ]
    
    for src, dst in geojson_files:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"Warning: {src} not found")

if __name__ == "__main__":
    print("Starting enhanced data conversion...")
    os.makedirs('public/data', exist_ok=True)
    
    if convert_excel_to_json():
        print("✓ Excel data converted successfully")
    else:
        print("✗ Failed to convert Excel data")
    
    copy_geojson_files()
    print("✓ GeoJSON files copied")
    
    print("\nData conversion completed!")
    print("Run 'npm run dev' to start the dashboard")