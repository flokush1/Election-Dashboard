# Critical Preprocessing Rules - Quick Reference

## 🎯 Most Important Alignment Rules

### 1. **Caste Handling for Unknown Values**

This is the **#1 cause** of prediction differences:

```python
# ✅ CORRECT (Now implemented in both systems)
raw_religion = "HINDU"  # or "MUSLIM", etc.
raw_caste = ""  # blank/NA/unknown

is_hindu = "HINDU" in raw_religion.upper()

if caste is blank/NA/unknown:
    if is_hindu:
        caste_tok = None  # DROP feature (don't add to categorical dict)
    else:
        caste_tok = "Caste_No_caste_system"  # Explicit token
```

```python
# ❌ WRONG (Old behavior - causes noise)
if caste is blank:
    caste_tok = "Caste_Obc"  # Always default to OBC
    # This was WRONG because Hindu unknown ≠ OBC
```

**Why this matters**: 
- Training data had missing castes for Hindus → model learned to ignore that feature
- Forcing a default injects false signal
- Non-Hindus genuinely have "no caste system" → needs explicit token

---

### 2. **Age <18 Handling**

```python
# ✅ CORRECT
age_int = 15  # under 18
if age_int < 18:
    age_group = "Age_18-25"  # Treat as youngest voter bracket
```

```python
# ❌ WRONG
if age_int < 18:
    age_group = "Age_60+"  # or None, or exclude
```

**Why**: Electoral roll shouldn't have <18, but if present, closest bucket is 18-25.

---

### 3. **Economic Category Text Variants**

```python
# ✅ CORRECT - Expanded mapping
econ_map = {
    'LOW INCOME': 'LOW INCOME AREAS',
    'LOWER MIDDLE': 'LOWER MIDDLE CLASS',
    'MIDDLE': 'MIDDLE CLASS',
    'UPPER MIDDLE': 'UPPER MIDDLE CLASS',
    'PREMIUM': 'PREMIUM AREAS',
    # Also codes:
    '1': 'LOW INCOME AREAS',
    '2': 'LOWER MIDDLE CLASS',
    '3': 'MIDDLE CLASS',
    '4': 'UPPER MIDDLE CLASS',
    '5': 'PREMIUM AREAS',
    # Short codes:
    'L': 'LOW INCOME AREAS',
    'M': 'MIDDLE CLASS',
    'H': 'PREMIUM AREAS',
}
```

**Why**: Excel data has inconsistent formatting.

---

### 4. **Locality vs locality**

```python
# ✅ CORRECT
mapped_voter['Locality'] = get_any(
    voter, 
    'Locality',  # Capital L first!
    'locality', 
    'Area', 
    'Location'
)
```

**Why**: Model was trained with capital `'Locality'` as the key. Case matters for DictVectorizer.

---

### 5. **MaleToFemaleRatio vs male_female_ratio**

```python
# ✅ CORRECT - Use the EXACT model feature name
# Model checkpoint has: 'MaleToFemaleRatio'
mapped_voter['MaleToFemaleRatio'] = get_any(
    voter,
    'MaleToFemaleRatio',  # Try exact first
    'male_female_ratio'    # Then alias
)
```

**But also** support alias in `numeric_alias_to_index`:
```python
self.numeric_alias_to_index = {
    'MaleToFemaleRatio': 3,
    'male_female_ratio': 3,  # Both map to same column index
}
```

---

### 6. **Blank vs "NA" vs None**

```python
# ✅ CORRECT - Treat all as missing
blank_strings = {"", "nan", "na", "n/a", "none", "null", "-"}

if str(value).strip().lower() in blank_strings:
    value = None  # or default
```

---

### 7. **Categorical Dict Structure**

```python
# ✅ CORRECT - Only include present features
cat = {}
if age_group is not None:
    cat["age"] = age_group
if caste_tok is not None:  # Might be None for Hindu-unknown
    cat["caste"] = caste_tok
if rel_tok is not None:
    cat["religion"] = rel_tok
# ... etc
```

```python
# ❌ WRONG - Force all keys
cat = {
    "age": age_group or "Age_18-25",
    "caste": caste_tok or "Caste_Obc",  # Bad!
    "religion": rel_tok or "Religion_Hindu",
}
```

**Why**: DictVectorizer treats missing keys as zeros (correct). Forcing defaults creates false positives.

---

### 8. **Numeric Default Values**

```python
# ✅ CORRECT - Use sensible defaults
{
    'land_rate_per_sqm': 0.0,  # Missing land rate → 0
    'construction_cost_per_sqm': 0.0,  # Missing cost → 0
    'population': 0.0,  # Missing pop → 0
    'MaleToFemaleRatio': 1.0,  # Missing ratio → 1.0 (balanced)
}
```

**Why**: After standardization (scaler), 0.0 becomes negative (below mean), which is appropriate. Ratio 1.0 is neutral.

---

## 🔍 Debugging Checklist

When predictions differ between systems:

### Step 1: Check Categorical Tokens
```python
print(f"Age: {age_group}")
print(f"Religion: {rel_tok}")
print(f"Caste: {caste_tok}")  # Check if None for Hindu-unknown
print(f"Economic: {econ_raw}")
print(f"Income: {income_tok}")
print(f"Locality: {locality}")
```

### Step 2: Check Numeric Values
```python
print(f"Land rate: {land_rate}")
print(f"Construction cost: {construction_cost}")
print(f"Population: {population}")
print(f"MF Ratio: {male_female_ratio}")
```

### Step 3: Check Feature Matrix
```python
X = predictor.preprocess_voter_data_vectorized([voter])
print(f"Shape: {X.shape}")
print(f"Non-zero count: {np.count_nonzero(X)}")
print(f"Feature names count: {len(predictor.feature_names)}")
```

### Step 4: Check DictVectorizer Output
```python
cat_dict = {
    "age": age_group,
    "religion": rel_tok,
    # ...
}
X_cat = predictor.vectorizer.transform([cat_dict])
print(f"Categorical features non-zero: {np.count_nonzero(X_cat)}")
```

---

## ⚡ Performance Tips

### Batch Processing
```python
# ✅ Good - Process multiple voters at once
voters = [voter1, voter2, voter3, ...]
results = predictor.predict_voters_vectorized(voters)
```

```python
# ❌ Slow - One by one
for voter in voters:
    result = predictor.predict_voters_vectorized([voter])
```

### Caching DictVectorizer
The vectorizer is already stored in the model checkpoint, so no need to recreate.

---

## 📊 Example Comparison

### Input
```json
{
  "name": "Rajesh Kumar",
  "age": 35,
  "religion": "Hindu",
  "caste": "",  // BLANK!
  "economic_category": "MIDDLE CLASS",
  "land_rate_per_sqm": 50000,
  "MaleToFemaleRatio": 1.05
}
```

### After Preprocessing (Both Systems Should Match)
```python
{
  "age": 35,
  "religion": "HINDU",
  "caste": "",  # Still blank
  "economic_category": "MIDDLE CLASS",
  "Locality": "",
  "land_rate_per_sqm": 50000.0,
  "MaleToFemaleRatio": 1.05,
  # ... other fields
}
```

### Categorical Dict for DictVectorizer
```python
{
  "age": "Age_26-35",
  "religion": "Religion_Hindu",
  # NO "caste" key! (because Hindu + unknown → drop)
  "economic": "MIDDLE CLASS",
  "income": "income_middle",
  "locality": ""
}
```

### Feature Matrix (partial view)
```
[0, 1, 0, 0, 0,  # age one-hot (Age_26-35 = 1)
 1, 0, 0, 0, 0,  # religion one-hot (Hindu = 1)
 0, 0, 0, 0, 0,  # caste one-hot (ALL ZEROS - dropped)
 0, 1, 0,        # economic one-hot (Middle = 1)
 0, 1, 0,        # income one-hot (middle = 1)
 ...,
 -0.5,           # land_rate (standardized)
 -0.3,           # construction_cost (standardized)
 -1.2,           # population (standardized)
 0.1]            # MF ratio (standardized)
```

### Prediction (Both Systems)
```json
{
  "predicted_party": "BJP",
  "party_probabilities": {
    "BJP": 0.42,
    "Congress": 0.28,
    "AAP": 0.18,
    "Others": 0.09,
    "NOTA": 0.03
  },
  "turnout_probability": 0.78
}
```

---

## 🎓 Training Data Context

Understanding the training data helps explain these rules:

1. **Caste for Hindus**: Many training records had blank caste for Hindu voters → model learned this is normal → we must replicate it

2. **Economic Categories**: Excel exports used various formats → model trained on canonical names → we must map to those

3. **Locality**: Specific neighborhood names were used as-is → case-sensitive match matters

4. **Numeric Ranges**: Standardization was based on training data distribution → we must use same scaler

---

## 🔒 Final Checklist

Before deploying, verify:

- [ ] `get_any()` and `to_float_safe()` are identical in both files
- [ ] Caste handling includes Hindu/non-Hindu logic
- [ ] Age <18 maps to 18-25
- [ ] Economic category uses expanded mapping
- [ ] `'Locality'` has capital L
- [ ] `'MaleToFemaleRatio'` matches model checkpoint
- [ ] Blank strings treated as None
- [ ] Categorical dict only includes non-None values
- [ ] Numeric defaults are appropriate (0.0 or 1.0)
- [ ] VoterPredictor inherits from same base class
- [ ] Model loading works without Streamlit dependencies

---

## 📞 Quick Fixes

### "Predictions differ by 5-10%"
→ Check caste handling for Hindu voters with blank caste

### "Feature matrix shape mismatch"
→ Check DictVectorizer feature_names vs model feature_names

### "All predictions are uniform"
→ Check if scaler is applied (numeric features should be standardized)

### "Turnout always 0.75"
→ Check if beta_T and alpha0 are loaded correctly

### "Party probs all equal"
→ Check if beta_P and gamma0 are loaded correctly
