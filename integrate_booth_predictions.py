"""
Generic script to integrate voter predictions with GeoJSON plot data for any booth.
Usage: python integrate_booth_predictions.py --booth 103 --assembly "New Delhi" --block "E" --locality "B.K.Dutt Colony"
"""

import pandas as pd
import json
import re
import argparse
import os
from pathlib import Path

def normalize_house_number(house_no):
    """Normalize house number for matching (remove spaces, hyphens, make uppercase)"""
    if pd.isna(house_no):
        return None
    house_str = str(house_no).strip().upper()
    # Remove common prefixes
    house_str = re.sub(r'^(HOUSE\s*NO\s*|H\s*NO\s*|BLOCK\s*)', '', house_str)
    # Normalize spaces and hyphens
    house_str = re.sub(r'[\s\-/]+', '', house_str)
    return house_str if house_str else None

def extract_booth_voters(predictions_csv, booth_no, assembly_name=None, locality=None, section_pattern=None):
    """
    Extract voters for a specific booth from the predictions CSV.
    
    Args:
        predictions_csv: Path to predictions CSV file
        booth_no: Booth number to filter
        assembly_name: Optional assembly name to filter
        locality: Optional locality name to filter
        section_pattern: Optional regex pattern to match section name
    """
    print(f"\n1. Loading voter predictions from {predictions_csv}...")
    predictions_df = pd.read_csv(predictions_csv)
    print(f"   ✓ Total records loaded: {len(predictions_df):,}")
    
    # Filter by booth number - check multiple possible column names
    booth_col = None
    if 'partno' in predictions_df.columns:
        booth_col = 'partno'
    elif 'booth_no.' in predictions_df.columns:
        booth_col = 'booth_no.'
    elif 'Booth_ID' in predictions_df.columns:
        booth_col = 'Booth_ID'
    
    if booth_col:
        booth_voters = predictions_df[predictions_df[booth_col] == booth_no].copy()
    else:
        # Filter by assembly and locality if no booth column
        booth_voters = predictions_df.copy()
        if assembly_name and 'assembly name' in predictions_df.columns:
            booth_voters = booth_voters[booth_voters['assembly name'].str.contains(assembly_name, case=False, na=False)]
        if locality and 'Locality' in predictions_df.columns:
            booth_voters = booth_voters[booth_voters['Locality'].str.contains(locality, case=False, na=False)]
        if section_pattern and 'section no & road name' in predictions_df.columns:
            booth_voters = booth_voters[booth_voters['section no & road name'].str.contains(section_pattern, case=False, na=False, regex=True)]
    
    print(f"   ✓ Filtered to {len(booth_voters)} voters for Booth {booth_no}")
    
    # Normalize house numbers for matching
    booth_voters['house_normalized'] = booth_voters['house number'].apply(normalize_house_number)
    
    return booth_voters

def load_geojson(geojson_path):
    """Load GeoJSON file"""
    print(f"\n2. Loading GeoJSON plot data from {geojson_path}...")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    print(f"   ✓ Loaded {len(geojson_data['features'])} plot features")
    return geojson_data

def match_voters_to_plots(geojson_data, voters_df, plot_field='plot_no.'):
    """
    Match voters to GeoJSON plots based on plot numbers.
    
    Args:
        geojson_data: GeoJSON FeatureCollection
        voters_df: DataFrame with voter predictions
        plot_field: Field name in GeoJSON properties for plot number
    """
    print(f"\n3. Matching voters to plots...")
    
    enhanced_features = []
    total_voters_matched = 0
    plots_with_data = 0
    unmatched_plots = []
    
    for feature in geojson_data['features']:
        props = feature['properties']
        plot_no = str(props.get(plot_field, '')).strip()
        plot_normalized = normalize_house_number(plot_no)
        
        # Try to match voters
        if plot_normalized:
            plot_voters = voters_df[voters_df['house_normalized'] == plot_normalized]
        else:
            plot_voters = pd.DataFrame()
        
        # Also try exact match on original plot number
        if len(plot_voters) == 0 and plot_no:
            plot_voters = voters_df[voters_df['house number'].astype(str).str.strip() == plot_no]
        
        if len(plot_voters) > 0:
            plots_with_data += 1
            total_voters_matched += len(plot_voters)
            
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
            props['voter_count'] = int(len(plot_voters))
            props['avg_turnout_prob'] = round(float(avg_turnout), 4)
            props['avg_prob_BJP'] = round(float(avg_bjp), 4)
            props['avg_prob_Congress'] = round(float(avg_congress), 4)
            props['avg_prob_AAP'] = round(float(avg_aap), 4)
            props['avg_prob_Others'] = round(float(avg_others), 4)
            props['avg_prob_NOTA'] = round(float(avg_nota), 4)
            props['predicted_winner'] = predicted_winner
            props['winner_probability'] = round(float(party_probs[predicted_winner]), 4)
            
            # Add individual voters with predictions
            props['voters'] = []
            for _, voter in plot_voters.iterrows():
                voter_data = {
                    'voter_id': voter['voters id'],
                    'name': voter['name'],
                    'age': int(voter['age']) if pd.notna(voter['age']) else None,
                    'gender': voter['gender'],
                    'religion': voter['religion'] if pd.notna(voter['religion']) else None,
                    'caste': voter['caste'] if pd.notna(voter['caste']) else None,
                    'economic': voter['economic_category'] if pd.notna(voter['economic_category']) else None,
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
            unmatched_plots.append(plot_no)
        
        enhanced_features.append(feature)
    
    print(f"   ✓ Matched {total_voters_matched} voters to {plots_with_data} plots")
    if unmatched_plots:
        print(f"   ℹ {len(unmatched_plots)} plots without voter data")
    
    return enhanced_features, total_voters_matched, plots_with_data

def save_enhanced_geojson(geojson_data, features, output_path, name):
    """Save enhanced GeoJSON with predictions"""
    enhanced_geojson = {
        "type": "FeatureCollection",
        "name": name,
        "crs": geojson_data.get('crs', {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        }),
        "features": features
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_geojson, f, indent=2, ensure_ascii=False)
    
    print(f"\n4. Output saved to: {output_path}")
    return enhanced_geojson

def print_statistics(voters_df, features):
    """Print aggregate statistics"""
    print("\n5. Aggregate Statistics:")
    print("   " + "-"*70)
    print(f"   Total Voters:                {len(voters_df):,}")
    print(f"   Average Turnout Probability: {voters_df['turnout_prob'].mean():.2%}")
    print(f"   Average BJP Support:         {voters_df['prob_BJP'].mean():.2%}")
    print(f"   Average Congress Support:    {voters_df['prob_Congress'].mean():.2%}")
    print(f"   Average AAP Support:         {voters_df['prob_AAP'].mean():.2%}")
    print(f"   Average Others Support:      {voters_df['prob_Others'].mean():.2%}")
    print(f"   Average NOTA:                {voters_df['prob_NOTA'].mean():.2%}")
    
    # Plot-level statistics
    plots_with_voters = [f for f in features if f['properties']['voter_count'] > 0]
    if plots_with_voters:
        print("\n6. Top Plots by Party Support:")
        print("   " + "-"*70)
        
        # Sort by BJP support
        bjp_plots = sorted(plots_with_voters, key=lambda x: x['properties']['avg_prob_BJP'], reverse=True)[:5]
        print("\n   Top 5 BJP Strong Plots:")
        for i, plot in enumerate(bjp_plots, 1):
            props = plot['properties']
            plot_id = props.get('plot_no.', props.get('PLOT_NO', 'Unknown'))
            print(f"   {i}. Plot {plot_id:10s} - {props['voter_count']:3d} voters - {props['avg_prob_BJP']:.1%} BJP support")
        
        # Sort by Congress support
        congress_plots = sorted(plots_with_voters, key=lambda x: x['properties']['avg_prob_Congress'], reverse=True)[:5]
        print("\n   Top 5 Congress Strong Plots:")
        for i, plot in enumerate(congress_plots, 1):
            props = plot['properties']
            plot_id = props.get('plot_no.', props.get('PLOT_NO', 'Unknown'))
            print(f"   {i}. Plot {plot_id:10s} - {props['voter_count']:3d} voters - {props['avg_prob_Congress']:.1%} Congress support")
        
        # Sort by AAP support
        aap_plots = sorted(plots_with_voters, key=lambda x: x['properties']['avg_prob_AAP'], reverse=True)[:5]
        print("\n   Top 5 AAP Strong Plots:")
        for i, plot in enumerate(aap_plots, 1):
            props = plot['properties']
            plot_id = props.get('plot_no.', props.get('PLOT_NO', 'Unknown'))
            print(f"   {i}. Plot {plot_id:10s} - {props['voter_count']:3d} voters - {props['avg_prob_AAP']:.1%} AAP support")

def main():
    parser = argparse.ArgumentParser(description='Integrate voter predictions with GeoJSON booth data')
    parser.add_argument('--booth', type=int, required=True, help='Booth number')
    parser.add_argument('--assembly', type=str, help='Assembly name (e.g., "New Delhi", "R K Puram")')
    parser.add_argument('--locality', type=str, help='Locality name (e.g., "B.K.Dutt Colony")')
    parser.add_argument('--block', type=str, help='Block identifier (e.g., "E", "C")')
    parser.add_argument('--geojson', type=str, required=True, help='Input GeoJSON file path')
    parser.add_argument('--predictions', type=str, default='predictions_new_delhi.csv', help='Predictions CSV file')
    parser.add_argument('--output', type=str, help='Output GeoJSON file path (auto-generated if not provided)')
    parser.add_argument('--plot-field', type=str, default='plot_no.', help='GeoJSON property field for plot number')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"BOOTH {args.booth} - VOTER PREDICTIONS INTEGRATION")
    if args.assembly:
        print(f"Assembly: {args.assembly}")
    if args.locality:
        print(f"Locality: {args.locality}")
    if args.block:
        print(f"Block: {args.block}")
    print("="*80)
    
    # Build section pattern for filtering
    section_pattern = None
    if args.block and args.locality:
        # e.g., "BLOCK-E.*B.?K.?DUTT"
        locality_pattern = args.locality.replace('.', r'\.?').replace(' ', r'\s*')
        section_pattern = f"BLOCK-?{args.block}.*{locality_pattern}"
    
    # Extract voters for this booth
    voters_df = extract_booth_voters(
        args.predictions,
        args.booth,
        args.assembly,
        args.locality,
        section_pattern
    )
    
    if len(voters_df) == 0:
        print("\n❌ ERROR: No voters found matching the criteria!")
        print("\nPlease check:")
        print("  - Booth number is correct")
        print("  - Assembly name matches data")
        print("  - Locality name matches data")
        return
    
    # Load GeoJSON
    geojson_data = load_geojson(args.geojson)
    
    # Match voters to plots
    enhanced_features, total_matched, plots_matched = match_voters_to_plots(
        geojson_data,
        voters_df,
        args.plot_field
    )
    
    if total_matched == 0:
        print("\n⚠️  WARNING: No voters were matched to plots!")
        print("\nThis might be due to:")
        print("  - Plot number format mismatch between GeoJSON and CSV")
        print("  - Incorrect plot field name specified")
        return
    
    # Generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        output_dir = 'public/data/geospatial'
        assembly_short = args.assembly.replace(' ', '') if args.assembly else 'Assembly'
        locality_short = args.locality.replace(' ', '').replace('.', '') if args.locality else 'Locality'
        block_part = f"Block{args.block}_" if args.block else ""
        output_path = f"{output_dir}/{assembly_short}_{block_part}Booth_{args.booth}_Plots_With_Predictions.geojson"
    
    # Save enhanced GeoJSON
    name_parts = []
    if args.assembly:
        name_parts.append(args.assembly)
    if args.block:
        name_parts.append(f"Block {args.block}")
    if args.locality:
        name_parts.append(args.locality)
    name_parts.append(f"Booth {args.booth} - Plots with Voter Predictions")
    name = ", ".join(name_parts)
    
    save_enhanced_geojson(geojson_data, enhanced_features, output_path, name)
    
    # Print statistics
    print_statistics(voters_df, enhanced_features)
    
    print("\n" + "="*80)
    print("✓ INTEGRATION COMPLETE!")
    print("="*80)
    
    # Sample output
    sample = next((f for f in enhanced_features if f['properties']['voter_count'] > 0), None)
    if sample:
        print("\nSample Enhanced Plot:")
        props = sample['properties']
        print(json.dumps({
            'plot_no': props.get('plot_no.', props.get('PLOT_NO', 'Unknown')),
            'block_no': props.get('block_no', 'Unknown'),
            'locality': props.get('loacalityn', 'Unknown'),
            'voter_count': props['voter_count'],
            'predicted_winner': props['predicted_winner'],
            'winner_probability': props['winner_probability'],
            'avg_turnout': props['avg_turnout_prob'],
            'party_support': {
                'BJP': props['avg_prob_BJP'],
                'Congress': props['avg_prob_Congress'],
                'AAP': props['avg_prob_AAP']
            },
            'sample_voter': props['voters'][0] if props['voters'] else None
        }, indent=2))

if __name__ == '__main__':
    main()
