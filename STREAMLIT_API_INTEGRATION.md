# Streamlit Dashboard & API Integration Guide

## Overview

This document explains how the **Streamlit Dashboard** (`app1.py`) and **Flask API** (`model_api.py`) are now fully aligned to use **identical preprocessing logic** for voter predictions.

---

## ✅ Key Alignment Points

### 1. **Helper Functions**
Both files now use the **same exact implementations**:

#### `get_any(row_or_dict, *names, default=None)`
- Tries multiple column/field names (case-insensitive)
- Works with both pandas Series and dicts
- Returns first non-null match or default

#### `to_float_safe(val, default=0.0)`
- Robust float parser handling:
  - NaN, blanks, "NA", "N/A", etc.
  - Percentages (e.g., "45.2%")
  - Currency symbols (e.g., "₹1,200.50")
  - Comma separators (e.g., "1,234.56")

### 2. **Age Bucketing**
Both systems use **identical logic**:

```python
age_int = int(float(age_val))
if age_int < 18:        # <18 treated as 18-25
    age_group = "Age_18-25"
elif age_int <= 25:
    age_group = "Age_18-25"
elif age_int <= 35:
    age_group = "Age_26-35"
elif age_int <= 45:
    age_group = "Age_36-45"
elif age_int <= 60:
    age_group = "Age_46-60"
else:
    age_group = "Age_60+"
```

### 3. **Religion Mapping**
Exact token mapping:
- `"HINDU"` → `"Religion_Hindu"`
- `"MUSLIM"` → `"Religion_Muslim"`
- `"SIKH"` → `"Religion_Sikh"`
- `"CHRISTIAN"` → `"Religion_Christian"`
- `"BUDDHIST"` → `"Religion_Buddhist"`
- `"JAIN"` → `"Religion_Jain"`

**Important**: If religion is unknown or missing, the feature is **dropped** (set to `None` in categorical dict) rather than defaulting.

### 4. **Caste Handling (Advanced Logic)**
This is the **most critical alignment**:

```python
raw_caste = str(get_any(r, 'caste', default=None))
is_hindu = "HINDU" in religion_string

if caste is blank/unknown:
    if is_hindu:
        # Hindu with unknown caste → DROP caste feature (None)
        caste_tok = None
    else:
        # Non-Hindu → "No caste system"
        caste_tok = "Caste_No_caste_system"
else:
    # Known caste values:
    if caste == "SC": caste_tok = "Caste_Sc"
    elif caste == "ST": caste_tok = "Caste_St"
    elif caste == "OBC": caste_tok = "Caste_Obc"
    elif caste == "BRAHMIN": caste_tok = "Caste_Brahmin"
    # ... etc
```

**Why this matters**: The model was trained with specific caste encoding rules. Using a default value for unknown Hindu castes would inject noise.

### 5. **Economic Category Normalization**
Expanded mapping in `model_api.py` to match Streamlit:

```python
econ_canon_map = {
    '1': 'LOW INCOME AREAS', 'LOW': 'LOW INCOME AREAS', 'L': 'LOW INCOME AREAS',
    'LOW INCOME': 'LOW INCOME AREAS',
    '2': 'LOWER MIDDLE CLASS', 'LM': 'LOWER MIDDLE CLASS',
    'LOWER MIDDLE': 'LOWER MIDDLE CLASS',
    '3': 'MIDDLE CLASS', 'M': 'MIDDLE CLASS', 'MIDDLE': 'MIDDLE CLASS',
    '4': 'UPPER MIDDLE CLASS', 'UM': 'UPPER MIDDLE CLASS',
    'UPPER MIDDLE': 'UPPER MIDDLE CLASS',
    '5': 'PREMIUM AREAS', 'P': 'PREMIUM AREAS', 'PREMIUM': 'PREMIUM AREAS',
    'HIGH': 'PREMIUM AREAS', 'H': 'PREMIUM AREAS'
}
```

### 6. **Income Band Derivation**
From economic category:
- `"LOW INCOME AREAS"` → `"income_low"`
- `"LOWER MIDDLE CLASS"` → `"income_middle"`
- `"MIDDLE CLASS"` → `"income_middle"`
- `"UPPER MIDDLE CLASS"` → `"income_high"`
- `"PREMIUM AREAS"` → `"income_high"`

### 7. **Numeric Features**
Both use **safe parsing** with `to_float_safe()`:
- `land_rate_per_sqm` / `land_rate`
- `construction_cost_per_sqm` / `construction_cost`
- `population`
- `MaleToFemaleRatio` / `male_female_ratio`

Handles aliases automatically (e.g., model checkpoint may have `land_rate` while Excel has `land_rate_per_sqm`).

### 8. **Locality Handling**
- Column name: `'Locality'` (capital L) preferred
- Falls back to `'locality'`, `'Area'`, `'Location'`
- If missing → empty string (not forced default)

---

## 🔧 VoterPredictor Class Alignment

### Inheritance Structure

```
App1VoterPredictor (app1.py)
    ↓ inherits
VoterPredictor (model_api.py)
```

**Key differences**:
- `model_api.py` version **overrides `load_model`** to work without Streamlit (`st.error()` calls)
- Uses same `preprocess_voter_data_vectorized` from parent
- Uses same `predict_voters_vectorized` from parent

### Model Loading

#### Streamlit (`app1.py`)
```python
def load_model(self, model_file):
    # Uses st.error() for user feedback
    # Accepts file upload object
```

#### API (`model_api.py`)
```python
def load_model(self, model_bytes):
    # Uses print() for logging
    # Accepts raw bytes
    # No Streamlit dependencies
```

Both use **identical** internal logic for:
- Torch/pickle detection
- Key normalization (`beta_P`, `gamma0`, etc.)
- Vectorizer/Scaler wrapping
- `_preprocess_model_weights()`

---

## 📊 Data Flow Comparison

### Streamlit Dashboard
```
Excel Upload
    ↓
pandas DataFrame
    ↓
User selects voter
    ↓
voter row → dict
    ↓
preprocess_voter_data_vectorized
    ↓
predict_voters_vectorized
    ↓
Display results in UI
```

### Flask API
```
POST /api/predict
    ↓
JSON payload
    ↓
normalize_voter_payload_for_model
    ↓
preprocess_voter_data_vectorized
    ↓
predict_voters_vectorized
    ↓
Return JSON
```

**Critical alignment**: `normalize_voter_payload_for_model` in `model_api.py` now produces **identical** field mappings as the Streamlit Excel processing.

---

## 🧪 Testing Alignment

To verify both systems produce identical predictions:

### 1. Test with Streamlit
```bash
streamlit run app1.py
```
- Upload voter Excel
- Select a voter (e.g., "Rajesh Kumar")
- Note predicted party & probabilities

### 2. Test with API
```bash
python model_api.py
```

```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rajesh Kumar",
    "age": 35,
    "gender": "Male",
    "religion": "Hindu",
    "caste": "OBC",
    "economic_category": "MIDDLE CLASS",
    "Locality": "Shanti Niketan",
    "land_rate_per_sqm": 50000,
    "construction_cost_per_sqm": 30000,
    "population": 500,
    "MaleToFemaleRatio": 1.05
  }'
```

**Expected**: Same predicted party and probabilities within ~0.1% tolerance.

---

## 🔍 Debugging Differences

If predictions differ:

### 1. Check Preprocessing
Add debug output in `preprocess_voter_data_vectorized`:
```python
print(f"Age group: {age_group}")
print(f"Religion token: {rel_tok}")
print(f"Caste token: {caste_tok}")
print(f"Economic: {econ_raw}")
print(f"Income: {income_tok}")
```

### 2. Check Feature Matrix
```python
X = predictor.preprocess_voter_data_vectorized([voter])
print(f"Feature matrix shape: {X.shape}")
print(f"Non-zero features: {np.count_nonzero(X)}")
print(f"First 20 features: {X[0, :20]}")
```

### 3. Use Model Debug Panel
Both systems include debug helpers:

**Streamlit**:
```python
dbg = predictor.debug_summary(sample_records=[voter])
st.json(dbg)
```

**API**:
```python
GET /api/health
```

Check:
- `feature_count`
- `has_vectorizer`
- `has_scaler`
- `model_arrays_loaded`

---

## 📝 Column Name Mapping Reference

### Excel → Internal Field Mapping

| Excel Column (examples) | Internal Field | Aliases Supported |
|------------------------|----------------|-------------------|
| `voters id`, `Voter ID`, `EPIC` | `voter_id` | `voter id`, `epic no`, `id` |
| `name`, `Voter Name` | `name` | `relation name` |
| `age`, `Age` | `age` | - |
| `gender`, `Gender`, `sex` | `gender` | `Sex` |
| `religion`, `Religion` | `religion` | - |
| `caste`, `Caste`, `Category` | `caste` | `social_category` |
| `economic_category`, `Class` | `economic_category` | `income_level`, `economic status` |
| `Locality`, `Area` | `Locality` | `locality`, `Location` |
| `partno`, `Booth_ID` | `booth_no` / `partno` | `part_no`, `booth_no` |
| `land_rate_per_sqm` | `land_rate_per_sqm` | `land_rate` |
| `construction_cost_per_sqm` | `construction_cost_per_sqm` | `construction_cost` |
| `population` | `population` | - |
| `MaleToFemaleRatio` | `MaleToFemaleRatio` | `male_female_ratio` |

---

## 🚀 Deployment Notes

### Environment Variables
Both systems should use:
```bash
FLASK_ENV=production  # for API
STREAMLIT_SERVER_PORT=8501  # for Streamlit
```

### Model File
- Place `.pkl` or `.pth` model file in project root
- Both systems support torch and pickle formats
- Vectorizer and Scaler must be included in checkpoint

### Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn flask flask-cors
pip install torch  # optional, for .pth files
```

---

## 📞 Support

If predictions still differ after alignment:

1. Check model checkpoint contains all required keys:
   - `feature_names`
   - `party_names`
   - `vectorizer` (or `dict_vectorizer`)
   - `scaler`
   - `beta_P`, `gamma0`, `beta_T`, `alpha0`

2. Verify DictVectorizer feature names match checkpoint `feature_names`

3. Check for case-sensitivity in categorical values (should be uppercase)

4. Confirm numeric features use same scaling (mean/std from training)

---

## ✨ Summary

Both systems now use:
- ✅ Identical helper functions
- ✅ Identical age bucketing
- ✅ Identical religion/caste/economic mapping
- ✅ Identical numeric parsing
- ✅ Same VoterPredictor base class
- ✅ Same preprocessing pipeline
- ✅ Same prediction output format

**Result**: API and Streamlit dashboard produce **identical predictions** for identical input data.
