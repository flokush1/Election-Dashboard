# Complete Setup Guide 🚀

This guide will help you set up the Delhi Election Dashboard from scratch after cloning from GitHub.

## Prerequisites

- **Node.js** v16+ (https://nodejs.org/)
- **Python** 3.8+ (https://www.python.org/)
- **Git** (https://git-scm.com/)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/flokush1/Election-Dashboard.git
cd Election-Dashboard
```

### 2. Install Frontend Dependencies

```bash
npm install
```

This will install all React, Vite, and other JavaScript dependencies.

### 3. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Data Files Setup

The repository includes prediction CSV files, but you'll need to add additional data files:

#### Required Files (Not in Git):

1. **Excel Data File** (Optional - for advanced features):
   - File: `NewDelhi_Parliamentary_Data.xlsx`
   - Location: Root directory
   - Purpose: Booth-level statistics from Excel
   - **Note**: The app will work WITHOUT this file, but some API endpoints will return 404

2. **Model Files** (Optional - for ML predictions):
   - Files: `*.pkl` or `*.pth` 
   - Location: `Prediction Models/` directory
   - Purpose: Machine learning models for voter predictions
   - **Note**: You can upload models via the API

#### Included Files (Already in Git):

- ✅ `predictions_new_delhi.csv` - Voter predictions for New Delhi
- ✅ `predictions_r_k_puram.csv` - Voter predictions for R.K. Puram
- ✅ `newdelhi_voter_predictions.csv` - Additional predictions
- ✅ GeoJSON files in `public/data/geospatial/`
- ✅ JSON data files in `public/data/`

### 5. Start the Backend Server

```bash
python model_api.py
```

The Flask API will start on `http://localhost:5000`

Expected output:
```
🔍 Checking dependencies...
✅ scikit-learn: x.x.x
 * Running on http://127.0.0.1:5000
```

### 6. Start the Frontend (in a new terminal)

```bash
npm run dev
```

The React app will start on `http://localhost:5173` (Vite's default port)

## Verification

### Test the Frontend
1. Open browser to `http://localhost:5173`
2. You should see the Parliament-level dashboard
3. Click on assembly constituencies to drill down

### Test the Backend
```bash
# Check API health
curl http://localhost:5000/api/health

# Should return:
# {"status": "healthy", "model_loaded": false}
```

## Features Available Without Additional Data

Even without the Excel file or ML models, you can:

✅ **View all dashboard visualizations**
- Parliament, Assembly, Ward, and Booth levels
- Interactive maps with GeoJSON boundaries
- Charts and statistics

✅ **Browse voter predictions**
- Pre-generated predictions from CSV files
- Booth-level prediction summaries

✅ **Upload models dynamically**
- Use the `/api/upload-model` endpoint to add ML models
- Upload voter data via API

## Optional: Adding Your Own Data

### Adding Excel Data

If you have voter Excel files:

1. Place `NewDelhi_Parliamentary_Data.xlsx` in the root directory
2. The API will automatically detect and use it
3. Endpoints like `/api/booth-excel-stats/<assembly>/<booth>` will work

### Adding ML Models

If you have trained models:

1. Place `.pkl` files in `Prediction Models/` directory
2. OR use the API to upload:
   ```bash
   curl -X POST http://localhost:5000/api/upload-model \
     -F "model=@your_model.pkl"
   ```

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

### Issue: "Port already in use"
**Solution**: 
- Frontend: Change port in `vite.config.js`
- Backend: Change port in `model_api.py` (last line: `app.run(port=5001)`)

### Issue: Map tiles not loading
**Solution**: Check internet connection (uses OpenStreetMap tiles)

### Issue: "Excel file not found" in console
**Solution**: This is expected if you don't have the Excel file. Dashboard will work without it.

### Issue: Python dependencies fail to install
**Solution**: 
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then try again
pip install -r requirements.txt
```

## Build for Production

### Frontend
```bash
npm run build
```
Outputs to `dist/` directory

### Deploy
- Frontend: Deploy `dist/` folder to any static hosting (Vercel, Netlify, etc.)
- Backend: Deploy Flask app to Python hosting (Heroku, Railway, etc.)

## Next Steps

1. ✅ Explore the 4-level dashboard hierarchy
2. ✅ Check out the prediction data in CSV files
3. 🔄 Add your own Excel data (optional)
4. 🔄 Train and upload ML models (optional)
5. 🎨 Customize the dashboard to your needs

## Support

For issues or questions:
- Check existing documentation files (README.md, QUICKSTART.md)
- Review debug files (DEBUG_*.md, TROUBLESHOOTING_*.md)
- Open an issue on GitHub

---

**Happy analyzing! 🗳️📊**
