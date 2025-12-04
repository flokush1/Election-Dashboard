#!/usr/bin/env python3
"""
Real ML Model API Server
Handles PKL model loading and voter predictions using app1.py's VoterPredictor
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import io
import json
import warnings
import traceback
import os
import re
import math
import sys
from werkzeug.utils import secure_filename

# Import VoterPredictor from app1.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app1 import VoterPredictor as App1VoterPredictor

# Check for required dependencies on startup
print("🔍 Checking dependencies...")
try:
    import sklearn
    print(f"✅ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn not found: {e}")
    print("Please install: pip install scikit-learn")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("⚠️ PyTorch not found (optional for some models)")

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ---------------------------------------------------------------------------
# CSV normalization helpers for resilient schema handling
# ---------------------------------------------------------------------------
def _norm_key(s):
    try:
        return re.sub(r"[^0-9a-z]+", "_", str(s).strip().lower()).strip("_")
    except Exception:
        return str(s)

def _build_colmap(df):
    """Return mapping of normalized name -> actual column name for a DataFrame."""
    colmap = {}
    for c in df.columns:
        colmap[_norm_key(c)] = c
    return colmap

def _find_col(colmap, aliases):
    """Find the first existing column by trying alias list (normalized)."""
    for a in aliases:
        key = _norm_key(a)
        if key in colmap:
            return colmap[key]
    return None

def _get_val(row, colmap, aliases, default=''):
    """Get a value from row using any of the aliases; returns default if missing/blank."""
    col = _find_col(colmap, aliases if isinstance(aliases, (list, tuple)) else [aliases])
    if not col:
        return default
    val = row.get(col, '')
    if val is None:
        return default
    sval = str(val).strip()
    return sval if sval != '' and sval.lower() != 'nan' else default

def _to_percent(val, default=0.0):
    """Parse number and normalize to percent scale (0-100). Accepts 0-1 or 0-100 inputs."""
    try:
        f = float(val)
        if math.isnan(f):
            return default
    except Exception:
        return default
    if f <= 1.0:
        return max(0.0, min(100.0, f * 100.0))
    return max(0.0, min(100.0, f))

# ---------------------------------------------------------------------------
# Assembly-specific CSV resolution helpers
# ---------------------------------------------------------------------------
def _to_slug(s: str) -> str:
    s = str(s or '').strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return s.strip("_")

def resolve_predictions_file(assembly_name: str) -> str:
    """Return path to predictions CSV for a given assembly.
    Prefers convention `predictions_<slug>.csv` if present, else falls back to legacy filenames.
    """
    slug = _to_slug(assembly_name)
    # Derive variants: normal slug, and a version that splits short tokens into letters (e.g., 'rk' -> 'r_k')
    base_tokens = [t for t in re.split(r"[^0-9a-z]+", str(assembly_name).strip().lower()) if t]
    letter_split_tokens = [('_'.join(list(t)) if t.isalpha() and 1 < len(t) <= 3 else t) for t in base_tokens]
    slug_variant_letter_split = '_'.join(letter_split_tokens)

    slug_variants = [slug]
    if slug_variant_letter_split and slug_variant_letter_split != slug:
        slug_variants.append(slug_variant_letter_split)

    # Try multiple candidates to support files with/without extension and both slug variants
    candidates = []
    for sv in slug_variants:
        candidates.append(f"predictions_{sv}.csv")
        candidates.append(f"predictions_{sv}")
    # Check both CWD and script directory for each candidate
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in candidates:
        if os.path.isabs(path):
            if os.path.exists(path):
                return path
        else:
            rel_path = path
            abs_path = os.path.join(script_dir, path)
            if os.path.exists(rel_path):
                return rel_path
            if os.path.exists(abs_path):
                return abs_path
    legacy_map = {
        'new_delhi': 'predictions_new_delhi.csv',
        'r_k_puram': 'predictions_r_k_puram.csv',
        'rk_puram': 'predictions_r_k_puram.csv',
    }
    # fallback to legacy mapping if available, otherwise first candidate
    fallback = legacy_map.get(slug)
    if fallback:
        # check cwd and script dir for fallback
        if os.path.exists(fallback):
            return fallback
        fb_abs = os.path.join(script_dir, fallback)
        if os.path.exists(fb_abs):
            return fb_abs
    # return the primary conventional name even if missing, so caller can report a clean error
    return candidates[0]

def _filter_df_by_assembly(df: pd.DataFrame, assembly_name: str) -> pd.DataFrame:
    """Filter dataframe rows to match assembly_name if an assembly column exists.
    Tries common column names; if none found, returns df unchanged.
    """
    colmap = _build_colmap(df)
    asm_col = _find_col(colmap, ['assembly name','AssemblyName','assembly_name','assembly','ac_name','AC','constituency'])
    if not asm_col:
        return df
    def norm_asm(s: str) -> str:
        s = str(s or '').strip().lower()
        # remove non-alphanumeric and collapse
        s = re.sub(r"[^0-9a-z]+", "", s)
        # unify common variants: 'rkpuram', 'r_k_puram', 'r.k. puram' -> 'rkpuram'
        return s
    try:
        target_norm = norm_asm(assembly_name)
        # Use 'contains' instead of exact match to handle formats like "44-R K PURAM PART NO : 38"
        series_norm = df[asm_col].astype(str).map(norm_asm)
        mask = series_norm.str.contains(target_norm, regex=False, na=False)
        return df[mask]
    except Exception:
        return df

def _booth_mask(series: pd.Series, booth_number: int) -> pd.Series:
    """Create a robust boolean mask matching booth_number against a pandas Series with
    potential values like '18', '018', '18.0', 'Booth 18', etc.
    """
    s = series.astype(str).str.strip()
    target = str(booth_number).strip()
    # direct string match
    eq_str = s == target
    # numeric match ignoring leading zeros/decimals and stray text
    def to_intish(x):
        try:
            m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(x))
            if not m:
                return None
            return int(float(m.group(0)))
        except Exception:
            return None
    ints = s.map(to_intish)
    eq_int = ints == int(booth_number)
    return eq_str | eq_int

# ---------------------------------------------------------------------------
# New Endpoint: Booth-level stats from NewDelhi_Parliamentary_Data.xlsx
# ---------------------------------------------------------------------------
@app.route('/api/booth-excel-stats/<assembly_name>/<int:booth_number>', methods=['GET'])
def booth_excel_stats(assembly_name, booth_number):
    """Aggregate booth-level stats directly from NewDelhi_Parliamentary_Data.xlsx.
    Returns vote share (per party), age groups, gender, religions, castes.
    """
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'NewDelhi_Parliamentary_Data.xlsx')
        if not os.path.exists(excel_path):
            return jsonify({'error': 'Parliament Excel file not found', 'path': excel_path}), 404

        # Load entire sheet (assumes manageable size). If large, optimize later.
        df = pd.read_excel(excel_path)

        # ✅ Use robust assembly matcher to handle messy Excel values
        df = _filter_df_by_assembly(df, assembly_name)

        # Build column map for finding booth column
        colmap = _build_colmap(df)

        # Detect booth column with expanded aliases
        booth_col = _find_col(colmap, [
            'booth', 'booth_no', 'booth_number', 'partno', 'part_no',
            'Booth_ID', 'Booth ID', 'booth_id', 'BoothNumber', 'boothno',
            'PartNo', 'Part No', 'Part_No', 'PARTNO'
        ])

        # ✅ Use robust booth matcher to handle "18", "018", "18.0", "Booth 18", etc.
        q = df
        if booth_col:
            mask = _booth_mask(df[booth_col], booth_number)
            q = df[mask]

        if q.empty:
            return jsonify({
                'error': 'No matching booth rows found',
                'assembly': assembly_name,
                'booth_number': booth_number,
                'debug': {
                    'total_rows_after_assembly_filter': len(df),
                    'booth_column_used': booth_col
                }
            }), 404

        # Take first row for percentages; use counts if provided
        row = q.iloc[0]

        def get_float(r, candidates, default=0.0):
            """Get float value from row using _get_val and _to_percent helpers"""
            val = _get_val(r, colmap, candidates, default='')
            if val == '':
                return default
            return _to_percent(val, default)

        # Party vote share (percent or counts)
        parties = {
            'BJP': get_float(row, ['BJP', 'bjp', 'BJP_share', 'BJP_ratio', 'BJP_votes']),
            'Congress': get_float(row, ['Congress', 'congress', 'INC', 'Congress_share', 'Congress_ratio', 'Congress_votes']),
            'AAP': get_float(row, ['AAP', 'aap', 'AAP_share', 'AAP_ratio', 'AAP_votes']),
            'Others': get_float(row, ['Others', 'others', 'Others_share', 'Others_ratio', 'Others_votes']),
            'NOTA': get_float(row, ['NOTA', 'nota', 'NOTA_share', 'NOTA_ratio', 'NOTA_votes'])
        }

        # Demographics
        age_groups = {
            '18-25': get_float(row, ['Age_18_25', '18-25', 'age 18-25', 'Age18_25']),
            '26-35': get_float(row, ['Age_26_35', '26-35', 'age 26-35', 'Age26_35']),
            '36-45': get_float(row, ['Age_36_45', '36-45', 'age 36-45', 'Age36_45']),
            '46-60': get_float(row, ['Age_46_60', '46-60', 'age 46-60', 'Age46_60']),
            '60+': get_float(row, ['Age_60_plus', '60+', 'age 60+', 'Age60_plus'])
        }

        genders = {
            'male': get_float(row, ['male', 'Male', 'gender_male']),
            'female': get_float(row, ['female', 'Female', 'gender_female'])
        }

        religions = {
            'hindu': get_float(row, ['Religion_Hindu', 'Hindu']),
            'muslim': get_float(row, ['Religion_Muslim', 'Muslim']),
            'sikh': get_float(row, ['Religion_Sikh', 'Sikh']),
            'christian': get_float(row, ['Religion_Christian', 'Christian']),
            'buddhist': get_float(row, ['Religion_Buddhist', 'Buddhist']),
            'jain': get_float(row, ['Religion_Jain', 'Jain']),
            'other': get_float(row, ['Religion_Other', 'Other'])
        }

        castes = {
            'sc': get_float(row, ['Caste_SC', 'SC']),
            'obc': get_float(row, ['Caste_OBC', 'OBC']),
            'brahmin': get_float(row, ['Caste_Brahmin', 'Brahmin']),
            'kshatriya': get_float(row, ['Caste_Kshatriya', 'Kshatriya']),
            'vaishya': get_float(row, ['Caste_Vaishya', 'Vaishya']),
            'st': get_float(row, ['Caste_ST', 'ST'])
        }

        # Optional total voters and turnout
        total_voters = get_float(row, ['Total_Voters', 'total_voters', 'Total'], 0)
        expected_turnout_pct = get_float(row, ['Turnout', 'Expected_Turnout', 'turnout_probability'], 0)

        return jsonify({
            'success': True,
            'assembly': assembly_name,
            'booth_number': booth_number,
            'total_voters': int(total_voters) if total_voters else None,
            'expected_turnout': int((expected_turnout_pct/100.0)*total_voters) if total_voters else None,
            'party_probabilities': parties,
            'expected_votes': parties,  # frontend can treat as vote share if percents
            'demographics': {
                'age_groups': age_groups,
                'genders': genders,
                'religions': religions,
                'castes': castes
            }
        })
    except Exception as e:
        print(f"Booth Excel stats error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Global API error handler to ensure JSON responses (prevents empty body fetch issues)
# ---------------------------------------------------------------------------
@app.errorhandler(500)
def handle_500(e):
    from flask import request
    path = request.path
    # Only intercept API paths; let default handler work for non-API routes
    if path.startswith('/api/'):
        import traceback
        return jsonify({
            'error': 'Internal server error',
            'exception': str(e),
            'trace_tail': traceback.format_exc().splitlines()[-5:]
        }), 500
    return e

def to_float_safe(val, default=0.0):
    """Robust float parser: handles NaN, blanks, commas, %, currency, stray text."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.number)):
        try:
            v = float(val)
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
        return default
    s = s.replace(",", "")  # remove thousands separators
    # handle percentages
    if s.endswith("%"):
        try:
            v = float(s[:-1]) / 100.0
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    # extract first numeric token (handles currency & stray letters like 'C', '₹1,200.50')
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    try:
        return float(m.group()) if m else default
    except Exception:
        return default

def get_any(row_or_dict, *names, default=None):
    """Fetch a value by trying multiple column/field names (case-insensitive)."""
    if isinstance(row_or_dict, pd.Series):
        data = row_or_dict.to_dict()
    else:
        data = dict(row_or_dict)
    # exact first
    for n in names:
        if n in data and pd.notna(data[n]):
            return data[n]
    # case-insensitive
    lower = {k.lower(): v for k, v in data.items()}
    for n in names:
        key = n.lower()
        if key in lower and pd.notna(lower[key]):
            return lower[key]
    return default

def normalize_voter_payload_for_model(voter_input, row_index_fallback=1):
    """
    Normalize an arbitrary voter JSON payload into the canonical fields
    expected by VoterPredictor.predict_voters_vectorized.

    This mirrors the mapping logic from /api/upload-voter-data so that
    /api/predict behaves consistently even if the frontend sends raw Excel-like
    columns (e.g. 'voters id', 'section no & road name', 'Locality', etc).
    """

    # local helpers (do NOT depend on DataFrame)
    def safe_int(val, default=0):
        try:
            if val is None:
                return default
            s = str(val).strip()
            if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
                return default
            return int(float(s))
        except Exception:
            return default

    def safe_float(val, default=0.0):
        try:
            if val is None:
                return default
            s = str(val).strip()
            if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
                return default
            return float(s)
        except Exception:
            return default

    # case-insensitive + partial-name lookup on a single dict
    def find_value(possible_names, default=None):
        if not isinstance(possible_names, (list, tuple)):
            possible_names = [possible_names]

        # exact (case-insensitive)
        lower_map = {str(k).lower().strip(): v for k, v in voter_input.items()}
        for name in possible_names:
            key = str(name).lower().strip()
            if key in lower_map:
                v = lower_map[key]
                if v is None:
                    continue
                s = str(v).strip()
                if s != "" and s.lower() != "nan":
                    return s

        # partial match: name substring in key
        for name in possible_names:
            target = str(name).lower().strip()
            for k, v in voter_input.items():
                k_norm = str(k).lower().strip()
                if target in k_norm:
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s != "" and s.lower() != "nan":
                        return s

        return default

    # --- ECONOMIC CATEGORY NORMALIZATION (same mapping as upload-voter-data) ---
    econ_canon_map = {
        '1': 'LOW INCOME AREAS', 'LOW': 'LOW INCOME AREAS', 'L': 'LOW INCOME AREAS',
        '2': 'LOWER MIDDLE CLASS', 'LM': 'LOWER MIDDLE CLASS',
        '3': 'MIDDLE CLASS', 'M': 'MIDDLE CLASS', 'MID': 'MIDDLE CLASS', 'E': 'MIDDLE CLASS',
        '4': 'UPPER MIDDLE CLASS', 'UM': 'UPPER MIDDLE CLASS', 'UPPER': 'UPPER MIDDLE CLASS',
        '5': 'PREMIUM AREAS', 'P': 'PREMIUM AREAS', 'PREMIUM': 'PREMIUM AREAS',
        'HIGH': 'PREMIUM AREAS', 'H': 'PREMIUM AREAS'
    }

    raw_econ = find_value(
        ['economic_category', 'Economic Category', 'economic status',
         'Economic Status', 'economic_status', 'income_level', 'class', 'Class']
    )
    econ_code = find_value(['economic_category_code', 'econ_code', 'eco_code'])

    econ_val = (raw_econ or '').strip()
    if not econ_val and econ_code:
        econ_val = econ_code.strip()

    if not econ_val:
        econ_val = 'MIDDLE CLASS'

    econ_key = econ_val.upper().strip()
    econ_full = econ_canon_map.get(econ_key, econ_key)
    # if econ_full is still a short code that exists in map, expand it
    if len(econ_full) <= 3 and econ_full in econ_canon_map:
        econ_full = econ_canon_map[econ_full]

    # --- voter_id detection (simple version) ---
    voter_id = (
        find_value(['voter_id', 'voter id', 'voters id', 'VoterID', 'EPIC',
                    'epic no', 'epic number', 'id']) or
        voter_input.get('voter_id') or
        f'VOTER_{row_index_fallback:05d}'
    )

    # --- canonical mapped voter dict (matches upload-voter-data output) ---
    mapped_voter = {
        'voter_id': voter_id,
        'name': (
            find_value(['name', 'Name', 'voter_name', 'Voter Name', 'relation name']) or
            voter_input.get('name') or
            'Unknown'
        ),
        'age': safe_int(find_value(['age', 'Age']), 30),
        'gender': (find_value(['gender', 'Gender', 'sex', 'Sex']) or
                   voter_input.get('gender') or
                   'Unknown').upper(),
        'religion': (find_value(['religion', 'Religion']) or
                     voter_input.get('religion') or
                     'HINDU').upper(),
        'caste': (find_value(['caste', 'Caste', 'Category', 'category',
                              'Social Category', 'social_category']) or
                  voter_input.get('caste') or
                  'GENERAL').upper(),
        'economic_category': econ_full,
        'economic_category_code': econ_code or econ_key,
        'locality': (
            find_value(['Locality', 'locality', 'Area', 'area', 'Location', 'location']) or
            voter_input.get('locality') or
            'Unknown'
        ),
        'assembly': (
            find_value(['assembly name', 'assembly', 'Assembly',
                        'Constituency', 'AC', 'assembly_constituency', 'ac_name']) or
            voter_input.get('assembly') or
            'Unknown'
        ),
        'section_road': (
            find_value(['section no & road name', 'section_road']) or
            voter_input.get('section_road') or
            'Unknown'
        ),
        'full_address': (
            find_value(['full_address', 'full address', 'Full Address',
                        'complete_address', 'Complete Address',
                        'residential_address', 'Residential Address', 'address', 'Address']) or
            voter_input.get('full_address') or
            voter_input.get('address') or
            'Unknown'
        ),
        'partno': safe_int(find_value(['partno', 'PartNo', 'part_no', 'Part No',
                                       'Booth_ID', 'booth_id', 'booth_no', 'Booth_No']), row_index_fallback),
        'booth_no': safe_int(find_value(['booth_no', 'Booth_No', 'Booth_ID',
                                         'booth_id', 'partno', 'PartNo', 'part_no', 'Part No']), row_index_fallback),
        'land_rate_per_sqm': safe_float(find_value(['land_rate_per_sqm', 'land_rate']), 0.0),
        'construction_cost_per_sqm': safe_float(find_value(['construction_cost_per_sqm', 'construction_cost']), 0.0),
        'population': safe_float(find_value(['population', 'Population']), 0.0),
        'male_female_ratio': safe_float(find_value(['male_female_ratio', 'MaleToFemaleRatio']), 1.0),

        # optional family metadata (kept for completeness)
        'household_id': find_value(['household_id']) or voter_input.get('household_id'),
        'family_id_main': find_value(['family_id_main']) or voter_input.get('family_id_main'),
        'core_family_id': find_value(['core_family_id']) or voter_input.get('core_family_id'),
        'core_family_head': find_value(['core_family_head']) or voter_input.get('core_family_head'),
        'family_head': find_value(['family_head']) or voter_input.get('family_head'),
        'core_family_size': safe_int(find_value(['core_family_size']), 1),
        'main_family_size': safe_int(find_value(['main_family_size']), 1),
        'family_by_chain': find_value(['family_by_chain']) or voter_input.get('family_by_chain'),
        'family_by_chain_id': find_value(['family_by_chain_id']) or voter_input.get('family_by_chain_id'),
        'house_number': find_value(['house number', 'house_number', 'houseno', 'house no']) or
                        voter_input.get('house_number') or 'Unknown',
        'relation_type': find_value(['relation type', 'relation_type']) or
                         voter_input.get('relation_type') or 'Unknown',
        'having_deleted_tag': find_value(['having deleted tag', 'having_deleted_tag']) or
                              voter_input.get('having_deleted_tag') or 'No',
        'houseno_base': find_value(['houseno_base']) or voter_input.get('houseno_base'),
        'houseno_normalized': find_value(['houseno_normalized']) or voter_input.get('houseno_normalized'),
        'addressbasekeynf': find_value(['addressbasekeynf']) or voter_input.get('addressbasekeynf'),
        'surname_effective': find_value(['surname_effective']) or voter_input.get('surname_effective'),
        'head_generation_level': safe_int(find_value(['head_generation_level']), 0),
    }

    # If booth_no ended up 0 but partno is nonzero, sync them
    if mapped_voter['booth_no'] == 0 and mapped_voter['partno'] != 0:
        mapped_voter['booth_no'] = mapped_voter['partno']

    return mapped_voter

# ---------------------------------------------------------------------------
# VoterPredictor wrapper - delegates to app1.py for predictions
# ---------------------------------------------------------------------------
class VoterPredictor(App1VoterPredictor):
    """
    API wrapper around app1.py's VoterPredictor
    Adds API-specific properties (model_loaded, model_file_path) 
    while using app1's prediction logic
    """
    def __init__(self):
        super().__init__()
        self.model_loaded = False
        self.model_file_path = None
        self._last_preprocess_diag = {}
        self._last_prediction_diag = {}
        self.alignment_thresholds = {"core": 0.7, "leaning": 0.4}
    
    def load_model(self, model_bytes):
        """Load model from bytes (API endpoint compatibility)"""
        try:
            print("📦 Loading model from bytes...")
            model_data = pickle.loads(model_bytes)
            
            # Use parent class's load_model_dict method
            success = self.load_model_dict(model_data)
            
            if success:
                self.model_loaded = True
                print(f"✅ Model loaded successfully")
                print(f"   Features: {len(self.feature_names)}")
                print(f"   Parties: {self.party_names}")
                print(f"   Has vectorizer: {self.vectorizer is not None}")
                print(f"   Has scaler: {self.scaler is not None}")
                return True
            else:
                print("❌ Model loading failed")
                return False
                
        except Exception as e:
            print(f"❌ Model load error: {e}")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def predict_voter(self, voter_data):
        """
        Single voter prediction for API compatibility
        Wraps app1.py's predict_voters_vectorized to return API format
        """
        try:
            print(f"🔮 Predicting for voter: {voter_data.get('name', 'Unknown')} (ID: {voter_data.get('voter_id', 'Unknown')})")
            
            # Call parent class's vectorized prediction
            results = self.predict_voters_vectorized([voter_data])
            
            if not results or len(results) == 0:
                return None, "Prediction failed - no results"
            
            # Get first result (single voter)
            pred = results[0]
            
            # Extract data
            party_probs = pred.get('party_probabilities', {})
            predicted_party = pred.get('predicted_party', 'Unknown')
            turnout_prob = pred.get('turnout_probability', 0.5)
            
            # Calculate confidence
            if party_probs:
                confidence = max(party_probs.values())
            else:
                confidence = 0.0
            
            print(f"🎯 Predicted: {predicted_party} with {confidence*100:.1f}% confidence")
            
            # Determine confidence level
            if confidence > 0.7:
                confidence_level = "High"
            elif confidence > 0.5:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            # Alignment classification
            alignment_category = 'swing'
            if confidence >= self.alignment_thresholds.get('core', 0.7):
                alignment_category = 'core'
            elif confidence >= self.alignment_thresholds.get('leaning', 0.4):
                alignment_category = 'leaning'
            
            # Build prediction factors for UI
            age_val = voter_data.get('age', 0)
            try:
                age_int = int(float(age_val)) if age_val not in (None, '') else 0
                if 18 <= age_int <= 25:
                    age_group = "Age 18-25"
                elif 26 <= age_int <= 35:
                    age_group = "Age 26-35"
                elif 36 <= age_int <= 45:
                    age_group = "Age 36-45"
                elif 46 <= age_int <= 60:
                    age_group = "Age 46-60"
                else:
                    age_group = "Age 60+"
            except Exception:
                age_group = "Unknown age"
            
            religion = str(voter_data.get('religion', '')).strip().title()
            caste = str(voter_data.get('caste', '')).strip().title()
            econ_cat = str(voter_data.get('economic_category', '')).strip().upper()
            locality = str(voter_data.get('locality', '')).strip().title()
            
            # Build result in API format
            result = {
                'predicted_party': predicted_party,
                'party_probabilities': party_probs,
                'turnout_probability': turnout_prob,
                'confidence_level': confidence_level,
                'model_confidence': f"{confidence*100:.1f}%",
                'alignment_category': alignment_category,
                'alignment_party': predicted_party,
                'alignment_confidence': confidence,
                'prediction_factors': {
                    'primary': f"{religion} - {caste}" if caste else religion,
                    'secondary': econ_cat if econ_cat else "Economic Status",
                    'tertiary': f"{age_group} | {locality}" if locality else age_group
                }
            }
            
            return result, None
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return None, str(e)



# Global predictor instance
predictor = VoterPredictor()

# Global variables to store uploaded data for search
uploaded_raw_data = []
uploaded_mapped_data = []

# Global predictor instance
predictor = VoterPredictor()

# Global variables to store uploaded data for search
uploaded_raw_data = []
uploaded_mapped_data = []

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed model status"""
    global uploaded_raw_data, uploaded_mapped_data
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'model_file': predictor.model_file_path,
        'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
        'party_count': len(predictor.party_names) if predictor.party_names else 0,
        'has_vectorizer': predictor.vectorizer is not None,
        'has_scaler': predictor.scaler is not None,
        'model_arrays_loaded': {
            'beta_P': predictor._beta_P_array is not None,
            'gamma0': predictor._gamma0_array is not None,
            'booth_effects_P': predictor._booth_effects_P_array is not None
        },
        'data_status': {
            'uploaded_voters_count': len(uploaded_mapped_data),
            'has_raw_data': len(uploaded_raw_data) > 0,
            'sample_voter_ids': [v.get('voter_id', 'NO_ID') for v in uploaded_mapped_data[:5]] if uploaded_mapped_data else []
        }
    })

@app.route('/api/upload-voter-data', methods=['POST'])
def upload_voter_data():
    """Fast server-side Excel processing endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Security check
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({'success': False, 'error': 'Please upload an Excel file (.xlsx or .xls)'}), 400
        
        # File size check (50MB limit for server processing)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > 50:
            return jsonify({
                'success': False, 
                'error': f'File too large: {file_size_mb:.1f}MB. Server processing limited to 50MB. Please split the file or use frontend processing.'
            }), 400
        
        print(f"📊 Processing Excel file: {filename} ({file_size_mb:.1f}MB)")
        
        # Read Excel file directly into pandas (much faster than frontend parsing)
        # If sheet_name is provided (form or query), load that sheet; otherwise, load and combine all sheets
        try:
            requested_sheet = request.form.get('sheet_name') or request.args.get('sheet_name')
            if requested_sheet:
                df = pd.read_excel(file, engine='openpyxl', sheet_name=requested_sheet)
                df_list = [df]
                sheets_meta = [{'name': requested_sheet, 'rows': len(df)}]
                print(f"✅ Excel read successfully from sheet '{requested_sheet}': {len(df)} rows, {len(df.columns)} columns")
            else:
                all_sheets = pd.read_excel(file, engine='openpyxl', sheet_name=None)
                df_list = list(all_sheets.values())
                sheets_meta = [{'name': name, 'rows': len(sdf)} for name, sdf in all_sheets.items()]
                if not df_list:
                    return jsonify({'success': False, 'error': 'No sheets found in Excel file'}), 400
                # Build a stable column order: start with first sheet's columns, then append new columns as they appear
                ordered_cols = list(df_list[0].columns)
                for sdf in df_list[1:]:
                    for c in sdf.columns:
                        if c not in ordered_cols:
                            ordered_cols.append(c)
                # Reindex each sheet to full ordered cols so concat keeps order
                normalized = [sdf.reindex(columns=ordered_cols) for sdf in df_list]
                df = pd.concat(normalized, ignore_index=True, sort=False)
                print(f"✅ Excel read successfully from all sheets: total {len(df)} rows across {len(df_list)} sheets, {len(df.columns)} columns")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to read Excel file: {str(e)}'}), 400
        
        # Convert to records (list of dicts) for frontend compatibility
        # Return ALL rows (user requested full dataset). Still constrained by 50MB file size check above.
        # If needed later, implement pagination instead of hard cap.
        max_rows = len(df)

        # Preserve original Excel column order
        # If we combined multiple sheets above, df.columns already follows the constructed ordered_cols
        original_column_order = list(df.columns)
        print(f"📋 Original Excel columns (in order): {original_column_order}")

        # Convert to records while preserving order
        raw_data = []
        for _, row in df.iterrows():
            # Create ordered dict to preserve column sequence
            row_dict = {}
            for col in original_column_order:
                val = row[col]
                # Handle NaN/empty values
                if pd.isna(val) or val == '':
                    row_dict[col] = ''
                else:
                    row_dict[col] = val
            raw_data.append(row_dict)

        # Also create normalized/mapped data for predictions
        mapped_data = []

        # Safe conversion functions
        def safe_int(val, default=0):
            """Safely convert value to int, handling floats and strings"""
            try:
                if val is None or val == '' or str(val).strip() == '':
                    return default
                # Convert to float first, then to int to handle '39.0' cases
                return int(float(val))
            except (ValueError, TypeError):
                return default
        
        def safe_float(val, default=0.0):
            """Safely convert value to float"""
            try:
                if val is None or val == '' or str(val).strip() == '':
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Pre-compute a probable voter id column if direct name variants fail
        voter_id_candidates = []
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            if non_null.empty:
                continue
            # Use string values
            sample_vals = non_null.astype(str).head(50)
            # Heuristic scoring: alphanumeric length >=8 and mostly uppercase/digits
            pattern_score = 0
            match_count = 0
            for v in sample_vals:
                v_strip = v.strip()
                if len(v_strip) >= 8 and re.fullmatch(r'[A-Za-z0-9]+', v_strip):
                    match_count += 1
            if len(sample_vals) > 0:
                pattern_score = match_count / len(sample_vals)
            uniqueness = non_null.nunique() / max(1, len(non_null))
            # Combined score
            total_score = pattern_score * 0.6 + uniqueness * 0.4
            voter_id_candidates.append((col, round(total_score, 4), round(pattern_score,3), round(uniqueness,3)))
        voter_id_candidates.sort(key=lambda x: x[1], reverse=True)
        detected_voter_id_col = None
        for col, score, _, _ in voter_id_candidates:
            norm = col.lower().strip()
            if norm in ['voter id','voters id','voterid','voter_id','epic','epic no','epic number','id'] or score >= 0.55:
                detected_voter_id_col = col
                break
        if detected_voter_id_col:
            print(f"🔎 Auto-detected voter ID column: {detected_voter_id_col}")
        else:
            print("⚠️ Could not confidently auto-detect voter ID column. Falling back to name variants per row.")

        id_debug_done = False

        for i, row in enumerate(df.itertuples()):
            row_dict = row._asdict()  # keys sanitized (spaces -> underscores) by pandas

            # Helper to get raw cell even if pandas sanitized column name
            def get_cell(col_name):
                if col_name in row_dict:
                    return row_dict[col_name]
                # Sanitize like pandas does for itertuples
                alt = re.sub(r'[^0-9a-zA-Z_]+', '_', col_name).strip('_')
                if alt in row_dict:
                    return row_dict.get(alt)
                return None

            # Helper to find column value (exact & partial, respecting sanitization)
            def find_value(possible_names, default=None):
                for name in possible_names:
                    target_norm = str(name).lower().strip()
                    for col in df.columns:
                        col_norm = str(col).lower().strip()
                        if col_norm == target_norm:
                            val = get_cell(col)
                            if val is None or val == '':
                                return default
                            sval = str(val).strip()
                            return sval if sval != '' else default
                        if target_norm in col_norm:  # partial
                            val = get_cell(col)
                            if val is None or val == '':
                                continue
                            sval = str(val).strip()
                            if sval != '':
                                return sval
                return default

            # Determine voter id for this row
            actual_voter_id = None
            # Priority: detected voter id column
            if detected_voter_id_col:
                cell_val = get_cell(detected_voter_id_col)
                if cell_val is not None and str(cell_val).strip() != '':
                    actual_voter_id = str(cell_val).strip()
            # Fallback explicit name variants
            if not actual_voter_id:
                actual_voter_id = find_value(['voters id', 'voter id', 'voter_id', 'VoterID', 'EPIC', 'epic no', 'epic number', 'ID'])

            # Row-wise EPIC/ID pattern fallback: scan all cell values in this row
            if not actual_voter_id or str(actual_voter_id).strip() == '' or str(actual_voter_id).lower() == 'nan':
                try:
                    for k, v in row_dict.items():
                        if v is None:
                            continue
                        s = str(v).strip()
                        if not s:
                            continue
                        # Prefer canonical EPIC-like pattern: 3 letters + 7 digits (e.g., IZM2521987)
                        m = re.search(r'\b[A-Za-z]{3}\d{7}\b', s)
                        if m:
                            actual_voter_id = m.group(0)
                            break
                        # Next, any long alphanumeric token ≥ 8 chars
                        m2 = re.search(r'\b[A-Za-z0-9]{8,}\b', s)
                        if m2:
                            actual_voter_id = m2.group(0)
                            break
                except Exception:
                    pass

            # Final fallback: generated id
            if not actual_voter_id or actual_voter_id.strip() == '' or actual_voter_id.lower() == 'nan':
                actual_voter_id = f'VOTER_{i+1:05d}'
                if i < 30:  # only spam first 30 missing
                    print(f"⚠️ Row {i+1}: No voter ID found, using {actual_voter_id}")
            else:
                if i < 30:
                    print(f"✅ Row {i+1}: Voter ID = {actual_voter_id}")

            # One-time debug of row_dict keys vs original columns
            if not id_debug_done and i == 0:
                print(f"🧪 First row sanitized keys: {list(row_dict.keys())[:25]}")
                print(f"🧪 Candidate voter id columns scored: {voter_id_candidates[:8]}")
                id_debug_done = True

            # Normalize economic category to full label for display/prediction factors
            econ_canon_map = {
                '1': 'LOW INCOME AREAS', 'LOW': 'LOW INCOME AREAS', 'L': 'LOW INCOME AREAS',
                '2': 'LOWER MIDDLE CLASS', 'LM': 'LOWER MIDDLE CLASS',
                '3': 'MIDDLE CLASS', 'M': 'MIDDLE CLASS', 'MID': 'MIDDLE CLASS', 'E': 'MIDDLE CLASS',
                '4': 'UPPER MIDDLE CLASS', 'UM': 'UPPER MIDDLE CLASS', 'UPPER': 'UPPER MIDDLE CLASS',
                '5': 'PREMIUM AREAS', 'P': 'PREMIUM AREAS', 'PREMIUM': 'PREMIUM AREAS', 'HIGH': 'PREMIUM AREAS', 'H': 'PREMIUM AREAS'
            }

            raw_econ = find_value(['economic_category', 'Economic Category', 'economic status', 'Economic Status', 'economic_status', 'income_level', 'class', 'Class'])
            econ_val = (raw_econ or 'MIDDLE CLASS').strip()
            econ_key = econ_val.upper().strip()
            econ_full = econ_canon_map.get(econ_key, econ_key)
            # If econ_full still looks like a short code, fallback to MIDDLE CLASS
            if len(econ_full) <= 3 and econ_full in econ_canon_map:
                econ_full = econ_canon_map[econ_full]

            mapped_voter = {
                'voter_id': actual_voter_id,
                'name': find_value(['name', 'Name', 'voter_name', 'Voter Name', 'relation name']) or 'Unknown',
                'age': safe_int(find_value(['age']), 30),
                'gender': (find_value(['gender', 'Gender', 'sex', 'Sex']) or 'Unknown').upper(),
                'religion': (find_value(['religion', 'Religion']) or 'HINDU').upper(),
                'caste': (find_value(['caste', 'Caste', 'Category', 'category', 'Social Category', 'social_category']) or 'GENERAL').upper(),
                'economic_category': econ_full,
                'economic_category_code': find_value(['economic_category_code', 'econ_code', 'eco_code']) or econ_key,
                'locality': find_value(['Locality', 'locality', 'Area', 'area', 'Location']) or 'Unknown',
                'assembly': find_value(['assembly name', 'assembly', 'Assembly', 'Constituency', 'AC', 'assembly_constituency', 'ac_name']) or 'Unknown',
                'section_road': find_value(['section no & road name']) or 'Unknown',
                'full_address': find_value(['full_address', 'full address', 'Full Address', 'complete_address', 'Complete Address', 'residential_address', 'Residential Address', 'address', 'Address']) or 'Unknown',
                'partno': safe_int(find_value(['partno', 'part_no']), i+1),
                'booth_no': safe_int(find_value(['partno', 'part_no', 'booth_no']), i+1),
                'land_rate_per_sqm': safe_float(find_value(['land_rate_per_sqm', 'land_rate']), 0.0),
                'construction_cost_per_sqm': safe_float(find_value(['construction_cost_per_sqm', 'construction_cost']), 0.0),
                'population': safe_float(find_value(['population']), 0.0),
                'male_female_ratio': safe_float(find_value(['male_female_ratio', 'MaleToFemaleRatio']), 1.0),
                'household_id': find_value(['household_id']) or None,
                'family_id_main': find_value(['family_id_main']) or None,
                'core_family_id': find_value(['core_family_id']) or None,
                'core_family_head': find_value(['core_family_head']) or None,
                'family_head': find_value(['family_head']) or None,
                'core_family_size': safe_int(find_value(['core_family_size']), 1),
                'main_family_size': safe_int(find_value(['main_family_size']), 1),
                'family_by_chain': find_value(['family_by_chain']) or None,
                'family_by_chain_id': find_value(['family_by_chain_id']) or None,
                'house_number': find_value(['house number']) or 'Unknown',
                'relation_type': find_value(['relation type']) or 'Unknown',
                'having_deleted_tag': find_value(['having deleted tag']) or 'No',
                'houseno_base': find_value(['houseno_base']) or None,
                'houseno_normalized': find_value(['houseno_normalized']) or None,
                'addressbasekeynf': find_value(['addressbasekeynf']) or None,
                'surname_effective': find_value(['surname_effective']) or None,
                'head_generation_level': safe_int(find_value(['head_generation_level']), 0),
            }
            mapped_data.append(mapped_voter)

        # NOTE: we include detection metadata for frontend debugging
        voter_id_detection_meta = {
            'detected_column': detected_voter_id_col,
            'candidates': [
                {'column': c, 'score': s, 'pattern_score': ps, 'uniqueness': u}
                for c, s, ps, u in voter_id_candidates[:10]
            ]
        }
        
        # Store data in global variables for search functionality
        global uploaded_raw_data, uploaded_mapped_data
        uploaded_raw_data = raw_data
        uploaded_mapped_data = mapped_data
        
        print(f"✅ ===== BACKEND DATA STORAGE SUCCESS =====")
        print(f"✅ Stored {len(mapped_data)} voters in global cache for search")
        print(f"✅ Sample voter IDs: {[v.get('voter_id', 'NO_ID') for v in mapped_data[:5]]}")
        print(f"✅ Global uploaded_mapped_data length: {len(uploaded_mapped_data)}")
        print(f"✅ First voter in cache: {uploaded_mapped_data[0] if uploaded_mapped_data else 'EMPTY'}")
        print(f"✅ ===== END BACKEND DATA STORAGE =====")
        
        return jsonify({
            'success': True,
            'raw_data': raw_data,
            'mapped_data': mapped_data,
            'total_rows': len(df),
            'columns': original_column_order,  # Preserve original Excel column order
            'rows_returned': max_rows,
            'sheets': sheets_meta,
            'file_info': {
                'name': filename,
                'size_mb': round(file_size_mb, 2)
            },
            'voter_id_detection': voter_id_detection_meta,
            'backend_cache_size': len(uploaded_mapped_data)  # Confirm backend stored data
        })
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error processing file: {str(e)}'
        }), 500

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    """Upload and load PKL model"""
    print("🔄 Model upload request received")
    try:
        if 'model' not in request.files:
            print("❌ No model file in request")
            return jsonify({'error': 'No model file provided'}), 400
        
        model_file = request.files['model']
        if model_file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📁 Received file: {model_file.filename}")
        
        # Safeguard: limit file size (e.g., 200MB) to prevent memory exhaustion
        model_file.seek(0, os.SEEK_END)
        size_bytes = model_file.tell()
        model_file.seek(0)
        print(f"📊 File size: {size_bytes / (1024*1024):.2f} MB")
        
        max_bytes = 200 * 1024 * 1024
        if size_bytes > max_bytes:
            error_msg = f'Model file too large ({size_bytes/1024/1024:.1f}MB). Limit {max_bytes/1024/1024}MB'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Read model data (binary)
        print("📖 Reading model data...")
        model_data = model_file.read()
        if not model_data:
            print("❌ Empty model data")
            return jsonify({'error': 'Uploaded file is empty'}), 400

        print(f"✅ Read {len(model_data)} bytes of model data")
        
        # Attempt load
        print("🔄 Loading model...")
        success, message = predictor.load_model(model_data)

        if success:
            print(f"✅ Model loaded successfully: {message}")
            return jsonify({
                'success': True,
                'message': message,
                'model_type': 'VoterPredictor',
                'feature_count': len(predictor.feature_names) if predictor.feature_names else 'Unknown',
                'file_size_mb': f"{size_bytes/1024/1024:.2f}"
            })
        else:
            print(f"❌ Model loading failed: {message}")
            return jsonify({
                'error': message,
                'file_size_mb': f"{size_bytes/1024/1024:.2f}",
                'hint': 'Ensure the pickle was created with compatible Python & library versions. For PyTorch, save state_dicts not full model objects.'
            }), 500

    except Exception as e:
        print(f"💥 Model upload error: {e}")
        traceback.print_exc()
        import traceback as tb
        error_response = {
            'error': str(e),
            'trace_tail': tb.format_exc().splitlines()[-8:],
            'suggestions': [
                'Verify the model was saved with Python version compatible with the server',
                'If torch model, prefer torch.save({"model_state_dict": model.state_dict(), ...})',
                'Avoid custom class pickles without the same class definitions on server',
                'Try re-saving with protocol=4 for compatibility'
            ]
        }
        print(f"📄 Error response: {error_response}")
        return jsonify(error_response), 500

@app.route('/api/predict', methods=['POST'])
def predict_voter():
    """Predict voter preference"""
    try:
        print(f"\n{'='*80}")
        print(f"🔮 PREDICTION REQUEST RECEIVED")
        print(f"{'='*80}")
        print(f"Model loaded: {predictor.model_loaded}")
        
        if not predictor.model_loaded:
            print(f"❌ Model not loaded. Status: {predictor.model_loaded}")
            return jsonify({'error': 'Model not loaded. Please upload a model first.'}), 400
        
        voter_data_raw = request.json
        if not voter_data_raw:
            return jsonify({'error': 'No voter data provided'}), 400
        
        print(f"\n📋 INCOMING VOTER DATA (RAW JSON):")
        print(f"   {json.dumps(voter_data_raw, indent=2)}")
        print(f"   Total keys in raw request: {len(voter_data_raw.keys())}")
        print(f"   All raw keys: {sorted(list(voter_data_raw.keys()))}")
        
        # 🔧 NEW: normalize raw payload into canonical model-ready structure
        mapped_voter = normalize_voter_payload_for_model(voter_data_raw, row_index_fallback=1)
        
        print(f"\n📋 NORMALIZED VOTER DATA (MAPPED FOR MODEL):")
        print(f"   {json.dumps(mapped_voter, indent=2)}")
        print(f"\n📋 Key field extraction (from mapped voter):")
        print(f"   Voter ID: {mapped_voter.get('voter_id', 'N/A')}")
        print(f"   Name: {mapped_voter.get('name', 'N/A')}")
        print(f"   Age: {mapped_voter.get('age', 'N/A')}")
        print(f"   Gender: {mapped_voter.get('gender', 'N/A')}")
        print(f"   Religion: {mapped_voter.get('religion', 'N/A')}")
        print(f"   Caste: {mapped_voter.get('caste', 'N/A')}")
        print(f"   Economic Category: {mapped_voter.get('economic_category', 'N/A')}")
        print(f"   Economic Code: {mapped_voter.get('economic_category_code', 'N/A')}")
        print(f"   Locality: {mapped_voter.get('locality', 'N/A')} ⚠️ CRITICAL FIELD")
        print(f"   Land Rate: {mapped_voter.get('land_rate_per_sqm', 'N/A')}")
        print(f"   Construction Cost: {mapped_voter.get('construction_cost_per_sqm', 'N/A')}")
        print(f"   Population: {mapped_voter.get('population', 'N/A')}")
        print(f"   Male/Female Ratio: {mapped_voter.get('male_female_ratio', 'N/A')}")
        print(f"   Booth / Part No: {mapped_voter.get('booth_no', 'N/A')} / {mapped_voter.get('partno', 'N/A')}")
        
        # Make prediction using normalized voter object
        result, error = predictor.predict_voter(mapped_voter)
        
        if error:
            print(f"❌ Prediction error: {error}")
            return jsonify({'error': error}), 500
        
        print(f"✅ Prediction successful")
        resp = {
            'success': True,
            'prediction': result,
            'mapped_voter': mapped_voter  # optional: helps debug in frontend
        }
        # Optional diagnostics for debugging mismatches
        if request.args.get('debug') == 'true':
            resp['diagnostics'] = {
                'preprocess': predictor._last_preprocess_diag,
                'prediction': predictor._last_prediction_diag,
                'feature_names_sample': predictor.feature_names[:10] if predictor.feature_names else [],
                'alignment_category': result.get('alignment_category'),
                'alignment_party': result.get('alignment_party'),
                'alignment_confidence': result.get('alignment_confidence')
            }
        return jsonify(resp)
        
    except Exception as e:
        print(f"Prediction API error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-voter', methods=['POST'])
def search_voter():
    """Search for a specific voter by ID in the uploaded data"""
    try:
        data = request.json
        search_id = data.get('voter_id', '').strip()
        
        if not search_id:
            return jsonify({'error': 'No voter ID provided'}), 400
        
        print(f"🔍 ===== BACKEND VOTER SEARCH =====")
        print(f"🔍 Searching for voter ID: {search_id}")
        
        # Search in both raw data and mapped data stored in global variables
        global uploaded_raw_data, uploaded_mapped_data
        
        print(f"🔍 Global cache status:")
        print(f"   - uploaded_mapped_data length: {len(uploaded_mapped_data) if uploaded_mapped_data else 0}")
        print(f"   - uploaded_raw_data length: {len(uploaded_raw_data) if uploaded_raw_data else 0}")
        
        if not uploaded_mapped_data:
            print(f"❌ No data in backend cache!")
            return jsonify({'error': 'No voter data uploaded yet. Please upload voter data first.'}), 400
        
        # Search for voter in mapped data with multiple strategies
        found_voter = None
        
        # Strategy 1: Exact match (case-insensitive)
        print(f"🔍 Strategy 1: Exact match (case-insensitive)")
        for voter in uploaded_mapped_data:
            voter_id = voter.get('voter_id', '').strip()
            if voter_id.upper() == search_id.upper():
                found_voter = voter
                print(f"✅ Found with exact match: {voter_id}")
                break
        
        # Strategy 2: Partial match
        if not found_voter:
            print(f"🔍 Strategy 2: Partial match")
            for voter in uploaded_mapped_data:
                voter_id = voter.get('voter_id', '').strip()
                if search_id.upper() in voter_id.upper() or voter_id.upper() in search_id.upper():
                    found_voter = voter
                    print(f"✅ Found with partial match: {voter_id}")
                    break
        
        # Strategy 3: Normalized match (remove special chars)
        if not found_voter:
            print(f"🔍 Strategy 3: Normalized match")
            normalized_search = re.sub(r'[^A-Z0-9]', '', search_id.upper())
            for voter in uploaded_mapped_data:
                voter_id = voter.get('voter_id', '').strip()
                normalized_voter_id = re.sub(r'[^A-Z0-9]', '', voter_id.upper())
                if normalized_voter_id == normalized_search:
                    found_voter = voter
                    print(f"✅ Found with normalized match: {voter_id}")
                    break
        
        if found_voter:
            print(f"✅ Found voter: {found_voter.get('name', 'Unknown')} (ID: {found_voter.get('voter_id')})")
            print(f"✅ ===== END BACKEND SEARCH =====")
            return jsonify({
                'success': True,
                'voter': found_voter,
                'found': True
            })
        else:
            # Show available voter IDs for debugging
            available_ids = [v.get('voter_id', '') for v in uploaded_mapped_data[:10]]  # First 10
            print(f"❌ Voter ID {search_id} not found. Available IDs sample: {available_ids}")
            print(f"❌ ===== END BACKEND SEARCH =====")
            return jsonify({
                'success': False,
                'error': f'Voter ID "{search_id}" not found in uploaded data',
                'found': False,
                'available_sample': available_ids,
                'total_voters_in_cache': len(uploaded_mapped_data)
            })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-voters', methods=['GET'])
def get_available_voters():
    """Get list of available voter IDs from uploaded data"""
    try:
        global uploaded_mapped_data
        
        if not uploaded_mapped_data:
            return jsonify({'error': 'No voter data uploaded yet. Please upload voter data first.'}), 400
        
        # Get first 1000 voter IDs to avoid massive response
        voter_ids = []
        for i, voter in enumerate(uploaded_mapped_data[:1000]):
            voter_ids.append({
                'voter_id': voter.get('voter_id', ''),
                'name': voter.get('name', 'Unknown'),
                'age': voter.get('age', 0),
                'gender': voter.get('gender', ''),
                'locality': voter.get('locality', '')
            })
        
        return jsonify({
            'success': True,
            'total_voters': len(uploaded_mapped_data),
            'shown_voters': len(voter_ids),
            'voters': voter_ids
        })
        
    except Exception as e:
        print(f"Available voters error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-family', methods=['POST'])
def predict_family():
    """Predict for voter's family members using core and chain family concepts from app.py"""
    try:
        print("🚀 Family prediction endpoint called!")
        
        if not predictor.model_loaded:
            print("❌ Model not loaded")
            return jsonify({'error': 'Model not loaded. Please upload a model first.'}), 400
        
        data = request.json
        print(f"📥 Received data keys: {list(data.keys()) if data else 'None'}")
        
        main_voter = data.get('voter')
        core_members = data.get('coreMembers', [])
        chain_members = data.get('chainMembers', [])
        core_family_id = data.get('core_family_id')
        chain_family_id = data.get('family_by_chain_id')
        
        print(f"👤 Main voter: {main_voter.get('name') if main_voter else 'None'}")
        print(f"🏠 Core members received: {len(core_members)}")
        print(f"🔗 Chain members received: {len(chain_members)}")
        print(f"🆔 Family IDs - Core: {core_family_id}, Chain: {chain_family_id}")
        
        if not main_voter:
            return jsonify({'error': 'No voter data provided'}), 400
        
        print(f"🔍 Finding family members for voter: {main_voter.get('name')}")
        
        results = []

        def _predict_member(member, family_type_label):
            try:
                pred_dict, err = predictor.predict_voter(member)
                if err:
                    print(f"⚠️ Prediction error for {family_type_label} member {member.get('name')}: {err}")
                    return None
                # Build response object aligned with frontend expectations
                return {
                    'name': member.get('name', 'Family Member'),
                    'voter_id': member.get('voter_id', ''),
                    'age': member.get('age', 0),
                    'family_type': family_type_label,
                    'predicted_party': pred_dict.get('predicted_party', 'Unknown'),
                    'party_probabilities': pred_dict.get('party_probabilities', {}),
                    'turnout_probability': pred_dict.get('turnout_probability', 0.0),
                    'confidence_level': pred_dict.get('confidence_level', 'Medium'),
                    'model_confidence': pred_dict.get('model_confidence', '0%')
                }
            except Exception as e:
                print(f"❌ Exception predicting {family_type_label} family member {member.get('name')}: {e}")
                return None

        # Process core family members
        if core_members:
            print(f"🏠 Processing {len(core_members)} core family members")
            for member in core_members[:5]:  # Limit to 5 members
                result_obj = _predict_member(member, 'core')
                if result_obj:
                    results.append(result_obj)

        # Process chain family members
        if chain_members:
            print(f"🔗 Processing {len(chain_members)} chain family members")
            for member in chain_members[:5]:  # Limit to 5 members
                result_obj = _predict_member(member, 'chain')
                if result_obj:
                    results.append(result_obj)
        
        print(f"✅ Generated predictions for {len(results)} family members")
        
        return jsonify({
            'success': True,
            'family_predictions': results
        })
        
    except Exception as e:
        print(f"Family prediction API error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-voter', methods=['POST'])
def debug_voter():
    """Debug endpoint to see exactly how a voter's features are processed"""
    try:
        if not predictor.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 400
            
        data = request.get_json()
        voter = data.get('voter', {})
        
        print(f"🔍 Debug processing voter: {voter}")
        
        # Show all available columns
        available_columns = list(voter.keys())
        print(f"📋 Available columns: {available_columns}")
        
        # Process single voter and show feature mapping
        X = predictor.preprocess_voter_data_vectorized([voter])
        
        # Get feature breakdown
        debug_info = {
            'available_columns': available_columns,
            'model_features': predictor.feature_names,
            'feature_count': len(predictor.feature_names),
            'processed_features': X[0].tolist(),
            'non_zero_features': np.count_nonzero(X[0]),
            'voter_data': voter,
            'model_loaded': predictor.model_loaded,
            'vectorizer_available': predictor.vectorizer is not None,
            'scaler_available': predictor.scaler is not None
        }
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        print(f"Debug API error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# New Endpoint: Parliament-level Excel Data Preview
# ---------------------------------------------------------------------------
@app.route('/api/parliament-data-preview', methods=['GET'])
def parliament_data_preview():
    """Return a lightweight preview (first N rows) of the New Delhi Parliamentary Excel file.
    Query params:
      limit (int, optional): number of rows (default 20, max 200)
    """
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'NewDelhi_Parliamentary_Data.xlsx')
        if not os.path.exists(excel_path):
            return jsonify({'error': 'Parliament Excel file not found', 'path': excel_path}), 404

        # Row limit handling
        limit = request.args.get('limit', default=20, type=int)
        limit = max(1, min(limit, 200))

        # Use pandas to read only needed portion
        # Read sheet names first for potential future expansion
        try:
            xls = pd.ExcelFile(excel_path)
            sheet_name = xls.sheet_names[0]
        except Exception:
            # Fallback single read
            sheet_name = 0

        # Read only first (limit) rows efficiently
        df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=limit)

        # Build column metadata
        columns_meta = []
        for col in df.columns:
            col_data = df[col]
            dtype = str(col_data.dtype)
            non_null = int(col_data.notna().sum())
            sample_val = None
            for v in col_data.head(5):
                if pd.notna(v):
                    sample_val = v
                    break
            columns_meta.append({
                'name': col,
                'dtype': dtype,
                'non_null': non_null,
                'sample': sample_val if sample_val is None or isinstance(sample_val, (str, int, float, bool)) else str(sample_val)
            })

        return jsonify({
            'success': True,
            'file': os.path.basename(excel_path),
            'sheet': sheet_name if isinstance(sheet_name, str) else 'Sheet1',
            'row_count_preview': len(df),
            'columns': columns_meta,
            'rows': df.fillna('').to_dict(orient='records')
        })
    except Exception as e:
        print(f"Parliament data preview error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/assembly-data-preview', methods=['GET'])
def assembly_data_preview():
    """Return preview rows filtered to a specific assembly.
    Query params:
      assembly (str, required): assembly name to filter (case-insensitive exact match)
      limit (int, optional): number of rows to return (default 20, max 300)
    Attempts to match assembly against common column name variants.
    """
    try:
        assembly = request.args.get('assembly')
        if not assembly:
            return jsonify({'error': 'assembly query parameter required'}), 400

        excel_path = os.path.join(os.path.dirname(__file__), 'NewDelhi_Parliamentary_Data.xlsx')
        if not os.path.exists(excel_path):
            return jsonify({'error': 'Parliament Excel file not found', 'path': excel_path}), 404

        limit = request.args.get('limit', default=20, type=int)
        limit = max(1, min(limit, 300))

        # Load only needed columns first for filtering (read a small chunk to detect columns)
        xls = pd.ExcelFile(excel_path)
        sheet_name = xls.sheet_names[0]
        sample_df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=25)
        cols = list(sample_df.columns)

        # Read entire sheet (still needed to filter). For very large sheets we could stream later.
        df_full = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Allow manual override via query param
        override_col = request.args.get('assembly_column')
        override_col_valid = override_col in df_full.columns if override_col else False

        # Gather candidate columns (textual first) and score them
        raw_candidates = []
        for c in df_full.columns:
            lc = str(c).lower()
            if ('assembly' in lc) or lc in {'ac','acname','ac_name'} or lc.startswith('ac_'):
                raw_candidates.append(c)

        # Always prefer explicit textual names if present
        preferred_order = []
        for key in ['AssemblyName','assembly_name','ASSEMBLY_NAME','assemblyname']:
            if key in df_full.columns:
                preferred_order.append(key)
        # De-duplicate while preserving order
        preferred_order = [c for i,c in enumerate(preferred_order) if preferred_order.index(c)==i]
        for c in raw_candidates:
            if c not in preferred_order:
                preferred_order.append(c)

        import re, difflib
        def normalize_candidate(s: str) -> str:
            s = str(s).lower().strip()
            s = re.sub(r'\s+', ' ', s)
            return s

        # Score each candidate column by exact & contains matches against requested assembly
        assembly_input = assembly
        assembly_norm_input = normalize_candidate(assembly_input)
        candidate_scores = []
        for c in preferred_order:
            series = df_full[c].astype(str).map(normalize_candidate)
            exact = int((series == assembly_norm_input).sum())
            contains = int(series.str.contains(re.escape(assembly_norm_input), regex=True).sum()) if assembly_norm_input else 0
            dtype = str(df_full[c].dtype)
            text_like = dtype == 'object'
            candidate_scores.append({
                'column': c,
                'dtype': dtype,
                'exact_matches': exact,
                'contains_matches': contains,
                'text_like': text_like
            })

        # Choose best column: max exact, then max contains, then text-like, then earlier order
        def sort_key(item):
            return (
                item['exact_matches'],
                item['contains_matches'],
                1 if item['text_like'] else 0,
                -preferred_order.index(item['column'])  # earlier order higher after reversing sign
            )
        candidate_scores_sorted = sorted(candidate_scores, key=sort_key, reverse=True)

        if override_col_valid:
            asm_col = override_col
            chosen_reason = 'override'
        elif candidate_scores_sorted:
            asm_col = candidate_scores_sorted[0]['column']
            chosen_reason = 'scored'
        else:
            return jsonify({'error': 'Could not detect assembly column in Excel file', 'available_columns': cols}), 500

        import re, difflib

        def normalize(s: str) -> str:
            s = str(s).strip().lower()
            s = re.sub(r'\s+', ' ', s)
            s = s.replace(' constituency', '').replace(' assembly', '')
            s = s.replace(' ac', '').replace('(ac)', '').strip()
            return s

        series_raw = df_full[asm_col].astype(str)
        series_norm = series_raw.map(normalize)
        assembly_norm = normalize(assembly)

        # Exact normalized match
        mask_exact = series_norm == assembly_norm
        filtered = df_full[mask_exact].head(limit)

        matched_exact_count = int(mask_exact.sum())

        # Contains match fallback (where either side contains the other) if no exact
        matched_contains_count = 0
        if filtered.empty:
            mask_contains = series_norm.str.contains(re.escape(assembly_norm), regex=True) | series_norm.apply(lambda v: assembly_norm in v)
            mask_contains &= series_norm.apply(lambda v: len(v) > 0)
            matched_contains_count = int(mask_contains.sum())
            filtered = df_full[mask_contains].head(limit)

        # Fuzzy match fallback if still empty
        fuzzy_used = False
        fuzzy_candidate = None
        fuzzy_ratio = 0.0
        if filtered.empty:
            unique_norm = list(series_norm.unique())
            # Use difflib to find closest
            close = difflib.get_close_matches(assembly_norm, unique_norm, n=1, cutoff=0.55)
            if close:
                fuzzy_candidate = close[0]
                fuzzy_used = True
                mask_fuzzy = series_norm == fuzzy_candidate
                fuzzy_ratio = difflib.SequenceMatcher(None, assembly_norm, fuzzy_candidate).ratio()
                filtered = df_full[mask_fuzzy].head(limit)

        # If still empty, surface sample unique values for debugging
        unique_sample = series_raw.head(200).unique().tolist()

        # Build column metadata for filtered subset
        columns_meta = []
        for col in filtered.columns:
            col_data = filtered[col]
            dtype = str(col_data.dtype)
            non_null = int(col_data.notna().sum())
            sample_val = None
            for v in col_data.head(5):
                if pd.notna(v):
                    sample_val = v
                    break
            columns_meta.append({
                'name': col,
                'dtype': dtype,
                'non_null': non_null,
                'sample': sample_val if sample_val is None or isinstance(sample_val, (str, int, float, bool)) else str(sample_val)
            })

        response = {
            'success': True,
            'assembly': assembly,
            'assembly_normalized': assembly_norm,
            'assembly_column_used': asm_col,
            'assembly_column_reason': chosen_reason,
            'row_count_preview': len(filtered),
            'columns': columns_meta,
            'rows': filtered.fillna('').to_dict(orient='records'),
            'debug': {
                'matched_exact_count': matched_exact_count,
                'matched_contains_count': matched_contains_count,
                'fuzzy_used': fuzzy_used,
                'fuzzy_candidate': fuzzy_candidate,
                'fuzzy_similarity': round(fuzzy_ratio, 3),
                'unique_sample_values': unique_sample[:30],
                'candidate_scores': candidate_scores_sorted,
                'override_requested': override_col,
                'override_used': override_col_valid
            }
        }
        return jsonify(response)
    except Exception as e:
        print(f"Assembly data preview error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict for multiple voters at once from uploaded data"""
    try:
        print("🚀 Batch prediction endpoint called!")
        
        if not predictor.model_loaded:
            print("❌ Model not loaded")
            return jsonify({'error': 'Model not loaded. Please upload a model first.'}), 400
        
        data = request.json
        voters = data.get('voters', [])
        
        if not voters:
            return jsonify({'error': 'No voter data provided'}), 400
        
        print(f"📊 Processing {len(voters)} voters for batch prediction")
        
        # Make predictions for all voters
        predictions = []
        successful = 0
        failed = 0
        
        for i, voter_data in enumerate(voters):
            try:
                result, error = predictor.predict_voter(voter_data)
                if error:
                    predictions.append({
                        'voter_id': voter_data.get('voter_id', f'voter_{i}'),
                        'name': voter_data.get('name', 'Unknown'),
                        'error': error,
                        'success': False
                    })
                    failed += 1
                else:
                    predictions.append({
                        'voter_id': voter_data.get('voter_id', f'voter_{i}'),
                        'name': voter_data.get('name', 'Unknown'),
                        'prediction': result,
                        'success': True
                    })
                    successful += 1
                    
                # Log progress for large batches
                if (i + 1) % 100 == 0:
                    print(f"📈 Progress: {i + 1}/{len(voters)} processed")
                    
            except Exception as e:
                predictions.append({
                    'voter_id': voter_data.get('voter_id', f'voter_{i}'),
                    'name': voter_data.get('name', 'Unknown'),
                    'error': str(e),
                    'success': False
                })
                failed += 1
        
        print(f"✅ Batch prediction completed: {successful} successful, {failed} failed")
        return jsonify({
            'success': True,
            'total_processed': len(voters),
            'successful': successful,
            'failed': failed,
            'predictions': predictions
        })
        
    except Exception as e:
        print(f"Batch prediction API error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/voter-preview', methods=['GET'])
def voter_preview():
    """
    Preview voter data from assembly files filtered by part number (booth)
    URL params: ?file=Complete_Voter_Data_NewDelhi.xlsx&partNo=1
    """
    try:
        file_name = request.args.get('file')
        part_no = request.args.get('partNo')
        
        if not file_name or not part_no:
            return jsonify({'error': 'Missing file or partNo parameter'}), 400
        
        # Construct the full file path
        file_path = os.path.join('VoterID_Data_Assembly', file_name)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_name}'}), 404
        
        print(f"Loading voter data from {file_path} for Part No: {part_no}")
        
        # Read only the necessary columns first to speed up loading
        try:
            # Try to read just the header first to identify columns
            header_df = pd.read_excel(file_path, nrows=0)
            all_columns = list(header_df.columns)
            
            # Find the part number column
            part_column = None
            possible_names = ['PartNo', 'Part No', 'partno', 'part_no', 'PARTNO', 'Part_No']
            
            for col_name in possible_names:
                if col_name in all_columns:
                    part_column = col_name
                    break
            
            if part_column is None:
                return jsonify({
                    'error': 'Part number column not found',
                    'available_columns': all_columns
                }), 400
            
            # Read the Excel file with optimizations
            df = pd.read_excel(file_path, dtype=str, na_filter=False)
            print(f"Loaded {len(df)} total rows from {file_name}")
            
        except Exception as read_error:
            return jsonify({
                'error': f'Failed to read Excel file: {str(read_error)}'
            }), 500
        
        # Convert part_no to string for comparison
        part_no_str = str(part_no)
        
        # Filter data by part number - use vectorized operation
        mask = df[part_column].astype(str) == part_no_str
        filtered_df = df[mask].copy()
        
        print(f"Found {len(filtered_df)} voters for Part No: {part_no}")
        
        if filtered_df.empty:
            return jsonify({
                'error': f'No data found for Part No: {part_no}',
                'total_parts_available': sorted(df[part_column].unique().tolist())[:20]  # Limit for performance
            }), 404
        
        # Prepare the response - limit to first 20 rows for faster loading
        columns = list(filtered_df.columns)
        
        # Convert to records for JSON serialization, limiting rows for preview
        preview_rows = min(20, len(filtered_df))
        preview_data = []
        
        for idx in range(preview_rows):
            row = filtered_df.iloc[idx]
            row_data = {}
            for col in columns:
                value = row[col]
                # Handle empty strings and NaN values
                if pd.isna(value) or value == '':
                    row_data[col] = None
                else:
                    row_data[col] = str(value).strip()
            preview_data.append(row_data)
        
        response_data = {
            'columns': columns,
            'preview': preview_data,
            'totalRows': len(filtered_df),
            'partNo': part_no,
            'fileName': file_name,
            'message': f'Found {len(filtered_df)} voters for Part No {part_no}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Voter preview error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/voter-predictions/<assembly_name>/<int:booth_number>', methods=['GET'])
def get_voter_predictions(assembly_name, booth_number):
    """Get voter predictions for a specific booth"""
    try:
        # Resolve predictions file strictly for this assembly
        predictions_file = resolve_predictions_file(assembly_name)
        
        asm_slug = _to_slug(assembly_name)
        print(f"Loading predictions for assembly: {assembly_name} (slug: {asm_slug})")
        print(f"Using predictions file: {predictions_file}")
        
        if not os.path.exists(predictions_file):
            print(f"Predictions file not found: {predictions_file}")
            return jsonify({'error': f'Predictions file not found for assembly: {assembly_name}'}), 404
        
        # Read data with optimizations
        df = pd.read_csv(
            predictions_file,
            dtype=str,
            na_filter=False
        )
        
        print(f"📊 Total rows in CSV: {len(df)}")

        colmap = _build_colmap(df)
        
        # Find booth/partno column directly (skip assembly filtering since CSV is assembly-specific)
        booth_col = _find_col(colmap, [
            'partno', 'PartNo', 'Part No', 'part no', 'Part_No', 'part_no', 'PARTNO',
            'Booth_ID', 'Booth ID', 'booth_id', 'booth', 'booth_no', 'Booth_No',
            'BoothNumber', 'booth_number', 'boothno'
        ])
        if not booth_col:
            return jsonify({'error': 'Booth/PartNo column not found in CSV'}), 400
        
        print(f"📊 Using booth column: {booth_col}")
        
        # Filter by booth (robust matching)
        mask = _booth_mask(df[booth_col], booth_number)
        booth_data = df[mask]
        
        print(f"📊 Found {len(booth_data)} voters for booth {booth_number}")
        
        if booth_data.empty:
            # Provide diagnostic sample to help debug mismatched column values
            sample_vals = df[booth_col].astype(str).head(10).tolist()
            return jsonify({'error': f'No predictions found for booth {booth_number}', 'booth_column': booth_col, 'sample_values': sample_vals}), 404
        
        # Convert to list of dictionaries for JSON response
        voters = []
        for _, row in booth_data.iterrows():
            r = row.to_dict()
            # Voter core fields with aliases
            voter = {
                'Voter_ID': _get_val(r, colmap, ['voters id', 'Voter_ID', 'Voter ID', 'voter_id', 'VoterID', 'epic', 'epic_no', 'epic no'], ''),
                'section_no_road_name': _get_val(r, colmap, ['section no & road name', 'section_no_road_name', 'section no and road name'], ''),
                'assembly_name': _get_val(r, colmap, ['assembly name', 'assembly_name', 'ac_name', 'assembly'], ''),
                'name': _get_val(r, colmap, ['name', 'voter_name'], ''),
                'relation_name': _get_val(r, colmap, ['relation name', 'relation_name', 'father name', 'father_name', 'husband name', 'husband_name'], ''),
                'house_number': _get_val(r, colmap, ['house number', 'house_number', 'houseno', 'house no'], ''),
                'Age': _get_val(r, colmap, ['Age', 'age'], ''),
                'gender': _get_val(r, colmap, ['gender', 'Gender', 'sex'], ''),
                'relation_type': _get_val(r, colmap, ['relation type', 'relation_type'], ''),
                'having_deleted_tag': _get_val(r, colmap, ['having deleted tag', 'having_deleted_tag'], ''),
                'address': _get_val(r, colmap, ['full address', 'full_address', 'address'], '') or _get_val(r, colmap, ['section no & road name', 'section_no_road_name'], ''),
                'Religion': _get_val(r, colmap, ['Religion', 'religion'], ''),
                'Caste': _get_val(r, colmap, ['Caste', 'caste', 'Category', 'category'], ''),
                'Locality': _get_val(r, colmap, ['Locality', 'locality', 'area', 'location'], ''),
                'Economic': _get_val(r, colmap, ['Economic', 'economic', 'economic_category', 'Economic Category'], ''),
                'predictions': {
                    'turnout_prob': _to_percent(_get_val(r, colmap, ['turnout_prob', 'turnout', 'turnout_probability'], 0)),
                    'prob_BJP': _to_percent(_get_val(r, colmap, ['prob_BJP', 'prob_bjp', 'bjp_prob', 'bjp'], 0)),
                    'prob_Congress': _to_percent(_get_val(r, colmap, ['prob_Congress', 'prob_congress', 'congress_prob', 'congress'], 0)),
                    'prob_AAP': _to_percent(_get_val(r, colmap, ['prob_AAP', 'prob_aap', 'aap_prob', 'aap'], 0)),
                    'prob_Others': _to_percent(_get_val(r, colmap, ['prob_Others', 'prob_others', 'others_prob', 'others'], 0)),
                    'prob_NOTA': _to_percent(_get_val(r, colmap, ['prob_NOTA', 'prob_nota', 'nota_prob', 'nota'], 0))
                }
            }
            voters.append(voter)
        
        return jsonify({
            'success': True,
            'booth_number': booth_number,
            'assembly_name': assembly_name,
            'total_voters': len(voters),
            'voters': voters
        })
        
    except Exception as e:
        print(f"Error loading voter predictions: {str(e)}")
        return jsonify({'error': f'Failed to load voter predictions: {str(e)}'}), 500


@app.route('/api/booth-statistics/<assembly_name>/<int:booth_number>', methods=['GET'])
def get_booth_statistics(assembly_name, booth_number):
    """Get aggregated statistics for a specific booth from CSV predictions"""
    try:
        # Resolve predictions file strictly for this assembly
        predictions_file = resolve_predictions_file(assembly_name)
        
        print(f"Loading booth statistics for assembly: {assembly_name}, booth: {booth_number}")
        print(f"Using predictions file: {predictions_file}")
        
        if not os.path.exists(predictions_file):
            return jsonify({'error': f'Predictions file not found for assembly: {assembly_name}'}), 404
        
        # Read data
        df = pd.read_csv(predictions_file, dtype=str, na_filter=False)
        colmap = _build_colmap(df)
        
        # Find booth/partno column directly (skip assembly filtering since CSV is assembly-specific)
        booth_col = _find_col(colmap, [
            'partno', 'PartNo', 'Part No', 'part no', 'Part_No', 'part_no', 'PARTNO',
            'Booth_ID', 'Booth ID', 'booth_id', 'booth', 'booth_no', 'Booth_No',
            'BoothNumber', 'booth_number', 'boothno'
        ])
        if not booth_col:
            return jsonify({'error': 'Booth/PartNo column not found in CSV'}), 400
        
        # Filter by booth number with robust mask
        mask = _booth_mask(df[booth_col], booth_number)
        booth_data = df[mask]
        
        if booth_data.empty:
            sample_vals = df[booth_col].astype(str).head(10).tolist()
            return jsonify({'error': f'No data found for booth {booth_number}', 'booth_column': booth_col, 'sample_values': sample_vals}), 404
        
        # Aggregate statistics
        total_voters = len(booth_data)
        
        # Calculate party probabilities (average across all voters)
        prob_bjp_list = []
        prob_congress_list = []
        prob_aap_list = []
        prob_others_list = []
        prob_nota_list = []
        turnout_prob_list = []
        
        for _, row in booth_data.iterrows():
            r = row.to_dict()
            prob_bjp_list.append(_to_percent(_get_val(r, colmap, ['prob_BJP', 'prob_bjp', 'bjp_prob', 'bjp'], 0), 0))
            prob_congress_list.append(_to_percent(_get_val(r, colmap, ['prob_Congress', 'prob_congress', 'congress_prob', 'congress'], 0), 0))
            prob_aap_list.append(_to_percent(_get_val(r, colmap, ['prob_AAP', 'prob_aap', 'aap_prob', 'aap'], 0), 0))
            prob_others_list.append(_to_percent(_get_val(r, colmap, ['prob_Others', 'prob_others', 'others_prob', 'others'], 0), 0))
            prob_nota_list.append(_to_percent(_get_val(r, colmap, ['prob_NOTA', 'prob_nota', 'nota_prob', 'nota'], 0), 0))
            turnout_prob_list.append(_to_percent(_get_val(r, colmap, ['turnout_prob', 'turnout', 'turnout_probability'], 0), 0))
        
        avg_prob_bjp = sum(prob_bjp_list) / len(prob_bjp_list) if prob_bjp_list else 0
        avg_prob_congress = sum(prob_congress_list) / len(prob_congress_list) if prob_congress_list else 0
        avg_prob_aap = sum(prob_aap_list) / len(prob_aap_list) if prob_aap_list else 0
        avg_prob_others = sum(prob_others_list) / len(prob_others_list) if prob_others_list else 0
        avg_prob_nota = sum(prob_nota_list) / len(prob_nota_list) if prob_nota_list else 0
        avg_turnout_prob = sum(turnout_prob_list) / len(turnout_prob_list) if turnout_prob_list else 0
        
        # Determine predicted winner
        party_probs = {
            'BJP': avg_prob_bjp,
            'Congress': avg_prob_congress,
            'AAP': avg_prob_aap,
            'Others': avg_prob_others,
            'NOTA': avg_prob_nota
        }
        predicted_winner = max(party_probs, key=party_probs.get)
        
        # Calculate expected votes based on turnout
        expected_turnout = int((avg_turnout_prob / 100) * total_voters)
        expected_votes = {
            'BJP': int((avg_prob_bjp / 100) * expected_turnout),
            'Congress': int((avg_prob_congress / 100) * expected_turnout),
            'AAP': int((avg_prob_aap / 100) * expected_turnout),
            'Others': int((avg_prob_others / 100) * expected_turnout),
            'NOTA': int((avg_prob_nota / 100) * expected_turnout)
        }
        
        # Calculate margin (difference between top 2)
        sorted_votes = sorted(expected_votes.values(), reverse=True)
        margin = sorted_votes[0] - sorted_votes[1] if len(sorted_votes) >= 2 else 0
        
        # Demographics breakdown
        age_groups = {'18-25': 0, '26-35': 0, '36-45': 0, '46-60': 0, '60+': 0}
        genders = {'male': 0, 'female': 0, 'other': 0}
        religions = {}
        castes = {}
        economic_categories = {}
        
        for _, row in booth_data.iterrows():
            r = row.to_dict()
            
            # Age groups
            age_str = _get_val(r, colmap, ['Age', 'age'], '0')
            try:
                age = int(age_str)
                if 18 <= age <= 25:
                    age_groups['18-25'] += 1
                elif 26 <= age <= 35:
                    age_groups['26-35'] += 1
                elif 36 <= age <= 45:
                    age_groups['36-45'] += 1
                elif 46 <= age <= 60:
                    age_groups['46-60'] += 1
                elif age > 60:
                    age_groups['60+'] += 1
            except:
                pass
            
            # Gender
            gender = _get_val(r, colmap, ['gender', 'Gender', 'sex'], '').lower()
            if 'male' in gender and 'female' not in gender:
                genders['male'] += 1
            elif 'female' in gender:
                genders['female'] += 1
            else:
                genders['other'] += 1
            
            # Religion
            religion = _get_val(r, colmap, ['Religion', 'religion'], 'Unknown')
            religions[religion] = religions.get(religion, 0) + 1
            
            # Caste
            caste = _get_val(r, colmap, ['Caste', 'caste'], 'Unknown')
            castes[caste] = castes.get(caste, 0) + 1
            
            # Economic category
            economic = _get_val(r, colmap, ['Economic', 'economic', 'economic_category'], 'Unknown')
            economic_categories[economic] = economic_categories.get(economic, 0) + 1
        
        # Get first row for booth metadata
        first_row = booth_data.iloc[0].to_dict()
        booth_address = _get_val(first_row, colmap, ['full address', 'full_address', 'address', 'section no & road name'], 'N/A')
        locality = _get_val(first_row, colmap, ['Locality', 'locality', 'area'], 'N/A')
        
        return jsonify({
            'success': True,
            'booth_number': booth_number,
            'assembly_name': assembly_name,
            'address': booth_address,
            'locality': locality,
            'total_voters': total_voters,
            'expected_turnout': expected_turnout,
            'avg_turnout_probability': round(avg_turnout_prob, 2),
            'predicted_winner': predicted_winner,
            'margin': margin,
            'party_probabilities': {
                'BJP': round(avg_prob_bjp, 2),
                'Congress': round(avg_prob_congress, 2),
                'AAP': round(avg_prob_aap, 2),
                'Others': round(avg_prob_others, 2),
                'NOTA': round(avg_prob_nota, 2)
            },
            'expected_votes': expected_votes,
            'demographics': {
                'age_groups': age_groups,
                'genders': genders,
                'religions': religions,
                'castes': castes,
                'economic_categories': economic_categories
            }
        })
        
    except Exception as e:
        print(f"Error loading booth statistics: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Failed to load booth statistics: {str(e)}'}), 500


@app.route('/api/voter-prediction/<voter_id>', methods=['GET'])
def get_individual_voter_prediction(voter_id):
    """Get prediction for a specific voter"""
    try:
        # Try to find the voter in both prediction files
        prediction_files = [
            'newdelhi_voter_predictions.csv',
            'rk_puram_voter_predictions.csv'
        ]
        
        voter_data = None
        used_file = None
        
        for predictions_file in prediction_files:
            if not os.path.exists(predictions_file):
                continue
                
            print(f"Searching for voter {voter_id} in {predictions_file}...")
            
            # Read data with optimizations
            df = pd.read_csv(
                predictions_file,
                dtype=str,
                na_filter=False
            )
            colmap = _build_colmap(df)
            voter_col = _find_col(colmap, ['Voter_ID', 'Voter ID', 'voter_id', 'epic', 'epic_no'])
            if not voter_col:
                continue
            # Filter by voter ID
            temp_voter_data = df[df[voter_col].astype(str).str.strip() == str(voter_id)]
            
            if not temp_voter_data.empty:
                voter_data = temp_voter_data
                used_file = predictions_file
                print(f"✅ Found voter in {predictions_file}")
                break
        
        if voter_data is None or voter_data.empty:
            print(f"❌ Voter {voter_id} not found in any prediction file")
            return jsonify({'error': f'No prediction found for voter {voter_id}'}), 404
        
        row = voter_data.iloc[0].to_dict()
        colmap = _build_colmap(voter_data)
        
        voter_prediction = {
            'Voter_ID': _get_val(row, colmap, ['Voter_ID', 'Voter ID', 'voter_id', 'epic', 'epic_no'], ''),
            'section_no_road_name': _get_val(row, colmap, ['section no & road name', 'section_no_road_name', 'section no and road name'], ''),
            'assembly_name': _get_val(row, colmap, ['assembly name', 'assembly_name', 'ac_name', 'assembly'], ''),
            'name': _get_val(row, colmap, ['name', 'voter_name'], ''),
            'relation_name': _get_val(row, colmap, ['relation name', 'relation_name', 'father name', 'father_name', 'husband name', 'husband_name'], ''),
            'house_number': _get_val(row, colmap, ['house number', 'house_number', 'houseno', 'house no'], ''),
            'Age': _get_val(row, colmap, ['Age', 'age'], ''),
            'gender': _get_val(row, colmap, ['gender', 'Gender', 'sex'], ''),
            'relation_type': _get_val(row, colmap, ['relation type', 'relation_type'], ''),
            'having_deleted_tag': _get_val(row, colmap, ['having deleted tag', 'having_deleted_tag'], ''),
            'address': _get_val(row, colmap, ['full address', 'full_address', 'address'], '') or _get_val(row, colmap, ['section no & road name', 'section_no_road_name'], ''),
            'Religion': _get_val(row, colmap, ['Religion', 'religion'], ''),
            'Caste': _get_val(row, colmap, ['Caste', 'caste', 'Category', 'category'], ''),
            'Locality': _get_val(row, colmap, ['Locality', 'locality', 'area', 'location'], ''),
            'Economic': _get_val(row, colmap, ['Economic', 'economic', 'economic_category', 'Economic Category'], ''),
            'Booth_ID': _get_val(row, colmap, ['Booth_ID', 'Booth ID', 'booth_id', 'booth', 'booth_no', 'Booth_No', 'partno', 'part no'], ''),
            'predictions': {
                'turnout_prob': _to_percent(_get_val(row, colmap, ['turnout_prob', 'turnout', 'turnout_probability'], 0)),
                'prob_BJP': _to_percent(_get_val(row, colmap, ['prob_BJP', 'prob_bjp', 'bjp_prob', 'bjp'], 0)),
                'prob_Congress': _to_percent(_get_val(row, colmap, ['prob_Congress', 'prob_congress', 'congress_prob', 'congress'], 0)),
                'prob_AAP': _to_percent(_get_val(row, colmap, ['prob_AAP', 'prob_aap', 'aap_prob', 'aap'], 0)),
                'prob_Others': _to_percent(_get_val(row, colmap, ['prob_Others', 'prob_others', 'others_prob', 'others'], 0)),
                'prob_NOTA': _to_percent(_get_val(row, colmap, ['prob_NOTA', 'prob_nota', 'nota_prob', 'nota'], 0))
            }
        }
        
        return jsonify({
            'success': True,
            'voter': voter_prediction
        })
        
    except Exception as e:
        print(f"Error loading voter prediction: {str(e)}")
        return jsonify({'error': f'Failed to load voter prediction: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting ML Model API Server...")
    print("📊 Upload your PKL model to /api/upload-model")
    print("🔮 Make predictions at /api/predict")
    print("👨‍👩‍👧‍👦 Family predictions at /api/predict-family")
    # Debug mode disabled to prevent auto-restart on file changes
    # Set debug=True if you want auto-reload during development
    app.run(debug=False, host='0.0.0.0', port=5000)
