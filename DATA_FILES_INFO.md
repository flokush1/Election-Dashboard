# Data Files Directory

This directory contains information about the data files needed for the Election Dashboard.

## Files Included in Git Repository ✅

The following files are already in the repository and will be available when you clone:

### 1. Prediction CSV Files
- `predictions_new_delhi.csv` - Voter predictions for New Delhi Assembly
- `predictions_r_k_puram.csv` - Voter predictions for R.K. Puram Assembly  
- `newdelhi_voter_predictions.csv` - Legacy prediction format

**Location**: Root directory  
**Size**: Small (< 10 MB)  
**Purpose**: Pre-computed voter party preference predictions  
**Format**: CSV with columns for voter demographics and predicted party

### 2. GeoJSON Boundary Files
- Parliament boundaries
- Assembly boundaries
- Ward boundaries  
- Booth-level geospatial data

**Location**: `public/data/geospatial/`  
**Purpose**: Map visualization and geographic analysis  
**Format**: GeoJSON

### 3. JSON Data Files
- `electoral-data.json` - Aggregated electoral statistics
- `summary-stats.json` - Summary statistics by level
- `sample-voters.json` - Sample voter data

**Location**: `public/data/`  
**Purpose**: Dashboard data visualization

## Files NOT Included in Git ❌

Due to size constraints and data sensitivity, the following files are excluded via `.gitignore`. The dashboard will work WITHOUT these files, but some features will be limited.

### 1. Excel Data Files

#### NewDelhi_Parliamentary_Data.xlsx
- **Purpose**: Complete voter roll with booth-level details
- **Size**: Large (> 50 MB)
- **Required for**: 
  - `/api/booth-excel-stats/<assembly>/<booth>` endpoint
  - Advanced booth analytics
  - Some data processing scripts
- **Without it**: 
  - Dashboard works fine
  - Predictions still available from CSV files
  - Some API endpoints return 404

**How to add**:
1. Obtain the Excel file from your data source
2. Place in root directory as `NewDelhi_Parliamentary_Data.xlsx`
3. Restart the backend server

**Alternative**: Upload voter data dynamically via `/api/upload-voter-data` endpoint

#### Other Excel Files
- `VoterID_Data_Assembly/*.xlsx` - Individual assembly Excel files
- **Purpose**: Assembly-specific data processing
- **Required for**: Some Python analysis scripts
- **Dashboard works without these**

### 2. Machine Learning Model Files

#### Model Formats
- `.pkl` - Scikit-learn pickle files
- `.pth` - PyTorch model files
- `.h5` - Keras/TensorFlow models (if used)

**Location**: `Prediction Models/<Assembly>/`  
**Example**: `Prediction Models/RK Puram/trained_electoral_model.pth`

**Purpose**: 
- Real-time voter prediction
- ML-based party preference analysis
- Turnout probability calculation

**Without it**: 
- Pre-computed predictions still available from CSV files
- `/api/predict` and `/api/predict-batch` endpoints won't work
- Can upload models dynamically

**How to add**:
1. Train your model using `app1.py` or similar
2. Save to `Prediction Models/<Assembly>/` directory
3. OR upload via API:
   ```bash
   curl -X POST http://localhost:5000/api/upload-model \
     -F "model=@your_model.pkl"
   ```

### 3. Compressed Archive Files

#### ZIP Files (*.zip)
- **Examples**: `VoterID_Data_Assembly.zip`, data backups
- **Size**: Very large (> 100 MB)
- **Purpose**: Compressed data backups or archives
- **Note**: These are excluded to avoid GitHub's file size limits
- **Without it**: Dashboard works fine - data already available in other formats

## Data File Checklist

Use this checklist to verify your setup:

### Minimum Setup (Dashboard works) ✅
- [x] `predictions_new_delhi.csv`
- [x] `predictions_r_k_puram.csv`
- [x] GeoJSON files in `public/data/geospatial/`
- [x] JSON files in `public/data/`

### Full Setup (All features enabled) 🚀
- [x] All minimum setup files
- [ ] `NewDelhi_Parliamentary_Data.xlsx`
- [ ] ML model files (`.pkl` or `.pth`)
- [ ] Optional: Additional assembly Excel files

## File Size Reference

```
Included in Git:
├── CSV files: ~5-10 MB
├── GeoJSON files: ~20-30 MB
├── JSON files: ~2-5 MB
└── Total: ~30-45 MB

NOT in Git:
├── Excel files: ~50-200 MB
├── Model files: ~10-100 MB
├── ZIP archives: ~100-300 MB
└── Total: ~160-600 MB
```

## Data Privacy Note

Excel files containing voter information are not included in the public repository to protect voter privacy and comply with data protection regulations. If you need access to the complete dataset:

1. Ensure you have proper authorization
2. Contact the project maintainer
3. Follow data protection guidelines
4. Keep sensitive files local (never commit to public repos)

## Getting Started

For a complete setup guide, see **[SETUP_GUIDE.md](../SETUP_GUIDE.md)** in the root directory.

**Quick start**: Clone the repo and run `npm install` + `pip install -r requirements.txt`. The dashboard will work with the included CSV and GeoJSON files!
