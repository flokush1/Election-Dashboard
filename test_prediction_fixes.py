"""
Test script to verify prediction fixes
Compares predictions before and after fixes
"""
import pickle
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the model
pkl_path = r"Prediction Models\Madipur\MADIPUR_trained_electoral_model.pkl"

print("="*80)
print("PREDICTION FIX VERIFICATION TEST")
print("="*80)

with open(pkl_path, 'rb') as f:
    model_data = pickle.load(f)

vectorizer = model_data['vectorizer']
scaler = model_data['scaler']
model_state = model_data['model_state_dict']
booth_id_to_idx = model_data['booth_id_to_idx']
party_names = model_data['party_names']

# Test voter: Rukhsar
test_voter = {
    'voter_id': 'IXE2422111',
    'name': 'RUKHSAR',
    'age': 23,
    'gender': 'FEMALE',
    'religion': 'MUSLIM',
    'caste': 'NO CASTE SYSTEM',
    'locality': 'RAGHUBIR NAGAR',
    'economic_category': 'LOW INCOME AREAS',
    'land_rate_per_sqm': 23280,
    'construction_cost_per_sqm': 3480,
    'population': 1462,
    'male_female_ratio': 1.156,
    'partno': 134
}

print("\n📋 TEST VOTER DATA:")
for key, val in test_voter.items():
    print(f"   {key:25s}: {val}")

# Step 1: Map features correctly
print("\n" + "="*80)
print("STEP 1: FEATURE MAPPING")
print("="*80)

# Age mapping
age = test_voter['age']
if age <= 25:
    age_cat = 'Age_18-25'
elif age <= 35:
    age_cat = 'Age_26-35'
elif age <= 45:
    age_cat = 'Age_36-45'
elif age <= 60:
    age_cat = 'Age_46-60'
else:
    age_cat = 'Age_60+'

# Religion mapping
religion_cat = f"Religion_{test_voter['religion'].capitalize()}"

# Caste mapping (Muslim with NO CASTE SYSTEM)
caste_cat = 'Caste_No_caste_system'

# Economic - MUST match exact vocabulary
economic_cat = test_voter['economic_category']
valid_econ = ['LOW INCOME AREAS', 'LOWER MIDDLE CLASS', 'MIDDLE CLASS', 'UPPER MIDDLE CLASS', 'PREMIUM AREAS']
if economic_cat not in valid_econ:
    print(f"⚠️ ERROR: Economic category '{economic_cat}' not in model vocabulary!")
    economic_cat = 'MIDDLE CLASS'

# Income derivation
if 'LOW INCOME' in economic_cat:
    income_cat = 'income_low'
elif 'UPPER MIDDLE' in economic_cat or 'PREMIUM' in economic_cat:
    income_cat = 'income_high'
else:
    income_cat = 'income_middle'

# Locality - MUST be uppercase
locality_cat = test_voter['locality'].upper()

# Create categorical dict for vectorizer
cat_dict = {
    'age': age_cat,
    'religion': religion_cat,
    'caste': caste_cat,
    'income': income_cat,
    'economic': economic_cat,
    'locality': locality_cat
}

print("\n✅ CATEGORICAL DICT FOR VECTORIZER:")
for key, val in cat_dict.items():
    print(f"   {key:15s}: {val}")

# Step 2: Check if vectorizer knows these features
print("\n" + "="*80)
print("STEP 2: VECTORIZER VALIDATION")
print("="*80)

vocab = vectorizer.vocabulary_
found_features = []
missing_features = []

for key, val in cat_dict.items():
    expected_feature = f"{key}={val}"
    if expected_feature in vocab:
        found_features.append(expected_feature)
        print(f"   ✅ Found: {expected_feature} (index {vocab[expected_feature]})")
    else:
        missing_features.append(expected_feature)
        print(f"   ❌ Missing: {expected_feature}")

if missing_features:
    print(f"\n⚠️ WARNING: {len(missing_features)} features not in model vocabulary!")
    print("\nAvailable similar features:")
    for missing in missing_features:
        prefix = missing.split('=')[0]
        similar = [f for f in vocab.keys() if f.startswith(prefix + '=')]
        print(f"\n   For {missing}:")
        for sim in similar[:5]:
            print(f"      - {sim}")
else:
    print(f"\n✅ ALL CATEGORICAL FEATURES FOUND IN MODEL!")

# Step 3: Vectorize
print("\n" + "="*80)
print("STEP 3: FEATURE VECTORIZATION")
print("="*80)

X_cat = vectorizer.transform([cat_dict])
if hasattr(X_cat, 'toarray'):
    X_cat_array = X_cat.toarray()
else:
    X_cat_array = X_cat

print(f"   Categorical vector shape: {X_cat_array.shape}")
print(f"   Non-zero features: {np.count_nonzero(X_cat_array[0])}")

# Step 4: Scale numerical features
print("\n" + "="*80)
print("STEP 4: NUMERICAL FEATURE SCALING")
print("="*80)

numerical_features = np.array([[
    test_voter['land_rate_per_sqm'],
    test_voter['construction_cost_per_sqm'],
    test_voter['population'],
    test_voter['male_female_ratio']
]])

print(f"\n   Raw numerical features:")
print(f"      land_rate:          {numerical_features[0][0]:.2f}")
print(f"      construction_cost:  {numerical_features[0][1]:.2f}")
print(f"      population:         {numerical_features[0][2]:.2f}")
print(f"      male_female_ratio:  {numerical_features[0][3]:.4f}")

scaled_numerical = scaler.transform(numerical_features)

print(f"\n   Scaled numerical features:")
print(f"      land_rate:          {scaled_numerical[0][0]:.4f}")
print(f"      construction_cost:  {scaled_numerical[0][1]:.4f}")
print(f"      population:         {scaled_numerical[0][2]:.4f}")
print(f"      male_female_ratio:  {scaled_numerical[0][3]:.4f}")

# Step 5: Combine features
print("\n" + "="*80)
print("STEP 5: COMBINE FEATURES")
print("="*80)

X = np.hstack([X_cat_array, scaled_numerical])
X_tensor = torch.FloatTensor(X)

print(f"   Combined shape: {X_tensor.shape}")
print(f"   Expected: (1, 45) - 41 categorical + 4 numerical")

if X_tensor.shape[1] != 45:
    print(f"   ⚠️ ERROR: Expected 45 features, got {X_tensor.shape[1]}")
else:
    print(f"   ✅ Correct feature count!")

# Step 6: Make prediction
print("\n" + "="*80)
print("STEP 6: MODEL PREDICTION")
print("="*80)

booth_id = f"{test_voter['partno']}_2025"
if booth_id not in booth_id_to_idx:
    print(f"   ❌ ERROR: Booth {booth_id} not found in model!")
    sys.exit(1)

booth_idx = booth_id_to_idx[booth_id]
print(f"   Booth ID: {booth_id}")
print(f"   Booth Index: {booth_idx}")

# Extract model parameters
alpha0 = model_state['alpha0']
beta_T = model_state['beta_T']
booth_effects_T = model_state['booth_effects_T']
gamma0 = model_state['gamma0']
beta_P = model_state['beta_P']
booth_effects_P = model_state['booth_effects_P']

# Turnout prediction
turnout_logit = alpha0 + torch.matmul(X_tensor.squeeze(), beta_T) + booth_effects_T[booth_idx]
turnout_prob = torch.sigmoid(turnout_logit).item()

print(f"\n   TURNOUT PREDICTION: {turnout_prob:.2%}")

# Party predictions
party_logits = gamma0 + torch.matmul(X_tensor.squeeze(), beta_P) + booth_effects_P[booth_idx]
party_probs = torch.softmax(party_logits, dim=0)

print(f"\n   PARTY VOTE SHARE PREDICTIONS:")
for i, party in enumerate(party_names):
    bar_length = int(party_probs[i].item() * 50)
    bar = '█' * bar_length
    print(f"      {party:12s}: {party_probs[i].item():6.2%} {bar}")

top_party_idx = torch.argmax(party_probs).item()
top_party = party_names[top_party_idx]
top_prob = party_probs[top_party_idx].item()

print(f"\n   PREDICTED PARTY: {top_party} ({top_prob:.2%})")

# Step 7: Diagnosis
print("\n" + "="*80)
print("STEP 7: PREDICTION DIAGNOSIS")
print("="*80)

print(f"\n   Expected for 23-year-old Muslim female from low-income area:")
print(f"      - High turnout (young voters engage)")
print(f"      - AAP/Congress preference (Muslim + low-income demographics)")
print(f"      - Lower BJP preference")

if turnout_prob > 0.8:
    print(f"   ✅ Turnout prediction looks correct")
else:
    print(f"   ⚠️ Turnout seems low for young voter")

if party_probs[2].item() > 0.4:  # AAP
    print(f"   ✅ AAP preference looks correct for demographic")
elif party_probs[1].item() > 0.3:  # Congress
    print(f"   ✅ Congress preference looks reasonable for demographic")
else:
    print(f"   ⚠️ Unexpected party preference pattern")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
