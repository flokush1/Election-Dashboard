# Delhi Election Dashboard 🗳️

> **✅ Ready to Clone & Use!** This repository includes all necessary prediction data (CSV) and geospatial files (GeoJSON). Clone, install dependencies, and start exploring the dashboard immediately. Optional Excel files and ML models can be added later for advanced features.

A comprehensive full-stack electoral analysis platform for the New Delhi Parliamentary constituency, featuring machine learning-powered voter prediction, interactive geospatial visualization, and hierarchical data exploration across 4 administrative levels: Parliament → Assembly → Ward → Booth.

## 🎯 Overview

This dashboard combines real-time voter analytics with ML-based party preference predictions to provide actionable insights for electoral campaigns. Built with modern web technologies and powered by scikit-learn models, it processes voter demographics, economic indicators, and historical patterns to forecast election outcomes at booth-level granularity.

## Features

### 🤖 Machine Learning Predictions
- **Voter Preference Prediction** - ML models predict BJP/Congress/AAP/Others/NOTA alignment
- **Turnout Probability** - Estimates likelihood of voter participation
- **Confidence Scoring** - High/Medium/Low confidence levels for predictions
- **Batch Processing** - Predict thousands of voters efficiently
- **Family Analysis** - Predict voting patterns for entire families
- **Model Upload** - Support for .pkl model files with scikit-learn/PyTorch

### 🏛️ Parliament Level
- Overall constituency overview with key statistics
- Interactive map of all assembly constituencies
- Party performance across the entire parliamentary seat
- Demographic breakdown and religious composition
- Economic category distribution

### 🗳️ Assembly Level  
- Detailed analysis of individual assembly constituencies
- Ward-wise breakdown within each assembly
- Age distribution and gender ratios
- Economic indicators and land rates
- Booth-wise performance metrics

### 🏘️ Ward Level
- Municipal ward-level analysis
- Polling booth distribution within wards
- Local demographic patterns
- Turnout and margin analysis

### 📊 Booth Level
- Individual polling booth details
- Detailed voter demographics
- Party-wise vote share with interactive charts
- Economic and social indicators
- Address and location information

## 🛠️ Technology Stack

### Frontend
- **React 18** - Modern UI framework with hooks
- **Vite** - Lightning-fast build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **React Leaflet** - Interactive maps with OpenStreetMap
- **Recharts** - Responsive data visualizations
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icon library
## 📂 Project Structure

```
delhi-election-dashboard/
├── src/                          # React frontend source
│   ├── components/
│   │   ├── levels/               # Parliament/Assembly/Ward/Booth views
│   │   ├── charts/               # Data visualization components
│   │   ├── stats/                # Statistics display widgets
│   │   ├── shared/               # Reusable UI components
│   │   └── InteractiveMap.jsx    # Leaflet map integration
│   ├── shared/
│   │   ├── utils.js              # Helper functions
│   │   └── dataProcessor.js      # Data transformation logic
│   ├── App.jsx                   # Main app component
│   └── main.jsx                  # React entry point
├── public/
│   └── data/                     # Static data files (JSON/GeoJSON)
├── model_api.py                  # Flask ML prediction API server
├── app1.py                       # Voter prediction ML logic
├── convert-data.py               # Excel → JSON conversion script
├── vite.config.js                # Vite build configuration
├── tailwind.config.js            # Tailwind CSS customization
└── package.json                  # Node dependencies
```

### Key Python Files
- `model_api.py` - REST API for voter predictions
- `app1.py` - Core ML model (VoterPredictor class)
- `app8.py` - Alternative prediction implementation
- `check_*.py` - Data validation scripts
- `analyze_*.py` - Data analysis utilities ├── components/
│   │   ├── ui/                   # Reusable UI components
│   │   ├── charts/               # Chart components
│   │   ├── stats/                # Statistics components
│   │   ├── levels/               # Level-specific dashboards
│   │   └── InteractiveMap.jsx    # Map component
│   ├── shared/
│   │   ├── utils.js              # Consolidated utility functions
│   │   └── dataProcessor.js      # Enhanced data processing logic
│   ├── App.jsx                   # Main application
│   └── main.jsx                  # Entry point
├── convert-data.py               # Enhanced data conversion script
└── package.json
```

## 🚀 Setup Instructions

### 📖 Quick Start Guide

**For detailed setup instructions, see [SETUP_GUIDE.md](./SETUP_GUIDE.md)**

**For data files information, see [DATA_FILES_INFO.md](./DATA_FILES_INFO.md)**

### Prerequisites
- **Node.js** v16+ (for frontend)
- **Python** 3.8+ (for ML backend)
- **npm** or **yarn** (package manager)

### Quick Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/flokush1/Election-Dashboard.git
   cd Election-Dashboard
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies:**
   ```bash
   # Recommended: Create virtual environment first
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   
   # Install from requirements file
   pip install -r requirements.txt
   ```

4. **Start the ML backend server:**
   ```bash
   python model_api.py
   ```
   The API will run on `http://localhost:5000`

5. **Start the frontend development server:**
   ```bash
   npm run dev
   ```
   The dashboard will open at `http://localhost:5173`

### Data Files

**✅ Included in repository:**
- CSV prediction files (`predictions_*.csv`)
- GeoJSON boundary files (`public/data/geospatial/`)
- JSON data files (`public/data/`)

**❌ NOT included (optional for full features):**
- Excel voter data: `NewDelhi_Parliamentary_Data.xlsx`
- ML model files: `.pkl`, `.pth` files

**The dashboard works without the optional files!** See [DATA_FILES_INFO.md](./DATA_FILES_INFO.md) for details.

## Dashboard Navigation

### 4-Level Navigation System

1. **Parliament Dashboard** (Default)
   - Overview of entire New Delhi Parliamentary constituency
   - Click on assembly constituencies to drill down

2. **Assembly Dashboard**  
   - Click on assembly constituency from parliament view
   - Shows detailed analysis of selected assembly
   - Click on wards to go deeper

3. **Ward Dashboard**
   - Click on ward from assembly view  
   - Municipal ward-level analysis
   - Click on individual booths for details

4. **Booth Dashboard**
   - Click on booth from ward view
   - Most detailed level with complete booth information
   - Full demographic and voting breakdown

### Interactive Features

- **Hover Effects**: Hover over map regions for quick stats
- **Click Navigation**: Click to drill down through levels
- **Breadcrumb Navigation**: Easy navigation back to higher levels
- **Responsive Charts**: Interactive charts with tooltips
- **Animated Transitions**: Smooth transitions between views

## Key Visualizations

### Charts & Graphs
- **Pie Charts**: Party vote share distribution
- **Bar Charts**: Booths won by party, age distribution
- **Horizontal Bar Charts**: Economic category distribution
- **Interactive Maps**: Color-coded by winner/performance

### Statistics Cards
- **Vote Count**: Total votes with formatting (K, M)
- **Booth Count**: Number of polling stations
- **Winner**: Leading party with color coding
- **Margins**: Victory margins and categories
- **Demographics**: Population, age, gender, religion

## Color Scheme

- **BJP**: Orange (`#FF9933`)
- **AAP**: Blue (`#0066CC`)  
- **Congress**: Green (`#00CC66`)
- **Others**: Gray (`#6B7280`)
- **NOTA**: Light Gray (`#9CA3AF`)

## Performance Features

- **Lazy Loading**: Components load as needed
- **Efficient Rendering**: Optimized re-renders
- **Responsive Design**: Works on desktop, tablet, mobile
- **Fast Navigation**: Smooth transitions between levels
- **Data Caching**: Processed data cached in memory

## Customization

### Adding New Visualizations
1. Create new chart component in `src/components/charts/`
2. Add to appropriate level dashboard
3. Update data processor if needed

### Modifying Styling
- Edit `tailwind.config.js` for theme changes
- Update color scheme in `src/shared/utils.js`
- Modify animations in component files

### Adding New Data Sources
1. Update `convert-data.py` to process new data
2. Modify `shared/dataProcessor.js` to handle new fields
3. Add visualizations in appropriate components

## Build for Production

```bash
npm run build
```

This creates a `dist/` folder with optimized production files.

## 📡 API Endpoints

The Flask backend (`model_api.py`) provides these endpoints:

### Model Management
- `GET /api/health` - Check API and model status
- `POST /api/upload-model` - Upload ML model (.pkl file)

### Voter Predictions
- `POST /api/predict` - Predict single voter
- `POST /api/predict-batch` - Predict multiple voters
- `POST /api/predict-family` - Predict family members
- `POST /api/search-voter` - Search voter by ID
- `GET /api/available-voters` - List uploaded voters

### Data Access
- `POST /api/upload-voter-data` - Upload Excel voter data
- `GET /api/booth-excel-stats/<assembly>/<booth>` - Get booth statistics
- `GET /api/voter-predictions/<assembly>/<booth>` - Get booth predictions
- `GET /api/parliament-data-preview` - Preview parliament data
- `GET /api/assembly-data-preview` - Preview assembly data

### Example API Call
```javascript
const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    voter_id: 'ABC1234567',
    name: 'Rajesh Kumar',
    age: 45,
    gender: 'MALE',
    religion: 'HINDU',
    caste: 'GENERAL',
    economic_category: 'MIDDLE CLASS',
    locality: 'Connaught Place'
  })
});
const data = await response.json();
console.log(data.prediction.predicted_party); // 'BJP'
```

## Browser Compatibility

- Chrome (recommended)
- Firefox  
- Safari
- Edge

## Troubleshooting

### Common Issues

1. **Map not loading**: Check internet connection for OpenStreetMap tiles
2. **Data not showing**: Ensure data files are converted and in public/data/
3. **Charts not responsive**: Check container dimensions and ResponsiveContainer usage

### Debug Mode
Enable console logging in components to track data flow and identify issues.

## 🧠 Machine Learning Model

The dashboard uses a custom `VoterPredictor` class that implements:

### Model Architecture
- **Feature Engineering** - 50+ features including demographics, economic indicators, geospatial data
- **Ensemble Methods** - Combines multiple models for robust predictions
- **Vectorization** - Efficient batch processing with TF-IDF for text features
- **Calibration** - Probability calibration for accurate confidence scores

### Input Features
- Demographics: Age, Gender, Religion, Caste
- Economic: Income level, Land rates, Construction costs
- Geographic: Locality, Assembly, Ward, Booth
- Family: Family size, relationships, household structure

### Output
- Party probabilities (BJP, Congress, AAP, Others, NOTA)
- Turnout probability (0-100%)
- Confidence level (High/Medium/Low)
- Alignment category (Core/Leaning/Swing)

### Model Training
Models are trained on historical election data with features extracted from:
- Voter rolls with demographic annotations
- Economic survey data
- Geographic information systems (GIS)
- Historical voting patterns

## 📊 Data Sources

This project analyzes electoral data from:
- **Election Commission of India** - Voter rolls and booth information
- **Census Data** - Demographic and economic indicators
- **GIS Systems** - Boundary shapefiles converted to GeoJSON
- **Custom Surveys** - Economic categorization and local insights

**Note:** Data files are not included in this repository due to size and privacy considerations.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open pull request

## License

This project is for educational and analysis purposes. Please ensure compliance with electoral data usage regulations.

---

**Dashboard Features Summary:**
✅ 4-level hierarchical navigation  
✅ Interactive maps with boundary visualization  
✅ Comprehensive demographic analysis  
✅ Party performance tracking  
✅ Economic indicator analysis  
✅ Responsive design  
✅ Real-time data processing  
✅ Smooth animations and transitions