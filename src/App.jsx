import React, { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import ParliamentLevel from './components/levels/ParliamentLevel';
import AssemblyLevel from './components/levels/AssemblyLevel';
import WardLevel from './components/levels/WardLevel';
import BoothLevel from './components/levels/BoothLevel';
import VoterPredictionPanel from './components/levels/VoterPredictionPanel';
import { processElectoralData, processGeoJsonData } from './shared/dataProcessor.js';
import { canonicalWardKey } from './shared/utils.js';

function App() {
  const [currentLevel, setCurrentLevel] = useState('parliament');
  const [selectedAssembly, setSelectedAssembly] = useState(null);
  const [selectedWard, setSelectedWard] = useState(null);
  const [selectedBooth, setSelectedBooth] = useState(null);
  
  const [data, setData] = useState(null);
  const [geoData, setGeoData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load electoral data
        const electoralResponse = await fetch('/data/electoral-data.json');
        let electoralData = [];
        
        if (electoralResponse.ok) {
          electoralData = await electoralResponse.json();
          console.log('Loaded electoral data:', electoralData.length, 'records');
        } else {
          console.warn('Electoral data not found, using mock data');
        }
        
        const processedData = processElectoralData(electoralData);
        setData(processedData);

        // Load GeoJSON boundaries
        const geoJsonFiles = {
          assembly: '/data/assembly-boundaries.geojson',
          ward: '/data/ward-boundaries.geojson',
          parliament: '/data/parliament-boundaries.geojson',
          booth: '/data/booth-boundaries.geojson'
        };

        const geoDataLoaded = {};
        
        for (const [level, url] of Object.entries(geoJsonFiles)) {
          try {
            let response = await fetch(url);
            // Fallback for booth boundaries: try known alternative file if the primary is missing
            if (!response.ok && level === 'booth') {
              const altUrl = '/data/geospatial/New_Delhi_Booth_Data.geojson';
              try {
                const altResp = await fetch(altUrl);
                if (altResp.ok) {
                  console.warn('booth boundaries not found at', url, '— using fallback', altUrl);
                  response = altResp;
                }
              } catch {}
            }

            if (response.ok) {
              const geoJson = await response.json();
              geoDataLoaded[level] = processGeoJsonData(geoJson, level);
              const featuresCount = Array.isArray(geoDataLoaded[level]?.features)
                ? geoDataLoaded[level].features.length
                : (Array.isArray(geoDataLoaded[level]) ? geoDataLoaded[level].length : 0);
              console.log(`Loaded ${level} boundaries:`, featuresCount, 'features');
            } else {
              console.warn(`${level} boundaries not found`);
              geoDataLoaded[level] = [];
            }
          } catch (err) {
            console.warn(`Error loading ${level} boundaries:`, err);
            geoDataLoaded[level] = [];
          }
        }
        
        setGeoData(geoDataLoaded);
        
        // Debug: Log available ward names
        console.log('🏘️ DEBUG: Available ward names in electoral data:', Object.keys(processedData.wards || {}));
        console.log('🏘️ DEBUG: Total wards:', Object.keys(processedData.wards || {}).length);
        
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err.message);
        // Use mock data as fallback
        const processedData = processElectoralData([]);
        setData(processedData);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const navigateToAssembly = (assemblyName) => {
    console.log('📍 navigateToAssembly called with:', assemblyName);
    console.log('📍 Available assemblies:', Object.keys(data.assemblies || {}));
    console.log('📍 Exact match exists?', data.assemblies && data.assemblies[assemblyName] ? 'YES' : 'NO');
    
    if (data.assemblies && data.assemblies[assemblyName]) {
      setSelectedAssembly(assemblyName);
      setSelectedWard(null);
      setSelectedBooth(null);
      setCurrentLevel('assembly');
      console.log('✅ Successfully navigated to assembly:', assemblyName);
    } else {
      console.log('❌ Assembly not found in data:', assemblyName);
      console.log('❌ Trying case-insensitive match...');
      
      // Try case-insensitive match
      let matchingKey = Object.keys(data.assemblies || {}).find(key => 
        key.toLowerCase() === assemblyName.toLowerCase()
      );
      
      // Fallback: try canonical key match (handles punctuation/spacing/aliases)
      if (!matchingKey) {
        const target = canonicalWardKey(assemblyName);
        matchingKey = Object.keys(data.assemblies || {}).find(key => canonicalWardKey(key) === target);
      }
      
      if (matchingKey) {
        console.log('✅ Found case-insensitive match:', matchingKey);
        setSelectedAssembly(matchingKey);
        setSelectedWard(null);
        setSelectedBooth(null);
        setCurrentLevel('assembly');
      } else {
        console.log('❌ No match found for:', assemblyName);
      }
    }
  };

  const navigateToWard = (wardName) => {
    console.log('🏘️ navigateToWard called with:', wardName);
    console.log('🏘️ Available wards:', Object.keys(data.wards || {}));
    console.log('🏘️ Exact match exists?', data.wards && data.wards[wardName] ? 'YES' : 'NO');
    
    if (data.wards && data.wards[wardName]) {
      setSelectedWard(wardName);
      setSelectedBooth(null);
      setCurrentLevel('ward');
      console.log('✅ Successfully navigated to ward:', wardName);
    } else {
      console.log('❌ Ward not found in data:', wardName);
      console.log('❌ Trying case-insensitive match...');
      
      // Try case-insensitive match
      let matchingKey = Object.keys(data.wards || {}).find(key => 
        key.toLowerCase() === wardName.toLowerCase()
      );
      
      // Fallback: try canonical key match (handles punctuation/spacing/aliases)
      if (!matchingKey) {
        const target = canonicalWardKey(wardName);
        matchingKey = Object.keys(data.wards || {}).find(key => canonicalWardKey(key) === target);
      }
      
      if (matchingKey) {
        console.log('✅ Found case-insensitive match:', matchingKey);
        setSelectedWard(matchingKey);
        setSelectedBooth(null);
        setCurrentLevel('ward');
      } else {
        console.log('❌ No match found for ward:', wardName);
        console.log('❌ Available ward keys sample:', Object.keys(data.wards || {}).slice(0, 10));
      }
    }
  };

  const navigateToBooth = (boothNumber) => {
    console.log('🏢 navigateToBooth called with:', boothNumber, 'in ward:', selectedWard);
    // Validate that booth exists in current ward
    if (selectedWard && data?.booths) {
      const wardBooth = data.booths.find(b => 
        Number(b.PartNo) === Number(boothNumber) && 
        b['Ward Name'] === selectedWard
      );
      if (wardBooth) {
        console.log('✅ Found booth in current ward:', wardBooth.PartNo, wardBooth['Ward Name']);
        setSelectedBooth(boothNumber);
        setCurrentLevel('booth');
      } else {
        console.log('❌ Booth not found in current ward, searching globally...');
        const globalBooth = data.booths.find(b => Number(b.PartNo) === Number(boothNumber));
        if (globalBooth) {
          console.log('⚠️ Found booth in different ward:', globalBooth['Ward Name']);
        }
        // Still navigate but add warning
        setSelectedBooth(boothNumber);
        setCurrentLevel('booth');
      }
    } else {
      setSelectedBooth(boothNumber);
      setCurrentLevel('booth');
    }
  };

  const navigateBack = () => {
    if (currentLevel === 'booth') {
      setSelectedBooth(null);
      setCurrentLevel('ward');
    } else if (currentLevel === 'ward') {
      setSelectedWard(null);
      setCurrentLevel('assembly');
    } else if (currentLevel === 'assembly') {
      setSelectedAssembly(null);
      setCurrentLevel('parliament');
    }
  };

  const navigateHome = () => {
    setSelectedAssembly(null);
    setSelectedWard(null);
    setSelectedBooth(null);
    setCurrentLevel('parliament');
  };

  const navigateToVoterPrediction = () => {
    setCurrentLevel('voter-prediction');
  };

  const getAvailableAssemblies = () => {
    if (!data?.assemblies) return [];
    return Object.keys(data.assemblies).sort();
  };

  const getAvailableWards = () => {
    if (!data?.wards || !selectedAssembly) return [];
    return Object.keys(data.wards)
      .filter(wardName => data.wards[wardName].assembly === selectedAssembly)
      .sort();
  };

  const getAvailableBooths = () => {
    if (!data?.booths || !selectedWard) return [];
    return data.booths
      .filter(booth => booth['Ward Name'] === selectedWard)
      .sort((a, b) => a.PartNo - b.PartNo);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Loading Delhi Election Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-gray-600">No data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <AnimatePresence mode="wait">
        {currentLevel === 'parliament' && (
          <ParliamentLevel
            key="parliament"
            data={data.parliament}
            assemblies={data.assemblies}
            geoData={geoData}
            onNavigateToAssembly={navigateToAssembly}
            onNavigateToVoterPrediction={navigateToVoterPrediction}
            availableAssemblies={getAvailableAssemblies()}
          />
        )}
        
        {currentLevel === 'assembly' && selectedAssembly && (
          <AssemblyLevel
            key={`assembly-${selectedAssembly}`}
            data={data.assemblies[selectedAssembly]}
            wards={data.wards}
            geoData={geoData}
            onNavigateToWard={navigateToWard}
            onNavigateBack={navigateBack}
            onNavigateHome={navigateHome}
            onNavigateToVoterPrediction={navigateToVoterPrediction}
            availableWards={getAvailableWards()}
            selectedAssembly={selectedAssembly}
            availableAssemblies={getAvailableAssemblies()}
            onAssemblyChange={navigateToAssembly}
          />
        )}
        
        {currentLevel === 'ward' && selectedWard && (
          <WardLevel
            key={`ward-${selectedWard}`}
            data={data.wards[selectedWard] || null}
            booths={data.booths}
            geoData={geoData}
            onNavigateToBooth={navigateToBooth}
            onNavigateBack={navigateBack}
            onNavigateHome={navigateHome}
            availableBooths={getAvailableBooths()}
            selectedWard={selectedWard}
            availableWards={getAvailableWards()}
            onWardChange={navigateToWard}
          />
        )}
        
        {currentLevel === 'ward' && selectedWard && !data.wards[selectedWard] && (
          <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
            <div className="text-center bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Ward Not Found</h2>
              <p className="text-gray-600 mb-4">No data available for ward: <strong>{selectedWard}</strong></p>
              <button 
                onClick={navigateBack}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Go Back
              </button>
            </div>
          </div>
        )}
        
        {currentLevel === 'booth' && selectedBooth && (
          <BoothLevel
            key={`booth-${selectedBooth}`}
            data={data.booths.find(b => 
              Number(b.PartNo) === Number(selectedBooth) && 
              (!selectedWard || b['Ward Name'] === selectedWard)
            )}
            geoData={geoData}
            onNavigateBack={navigateBack}
            onNavigateHome={navigateHome}
            availableBooths={getAvailableBooths()}
            selectedBooth={selectedBooth}
            onBoothChange={navigateToBooth}
          />
        )}

        {currentLevel === 'voter-prediction' && (
          <VoterPredictionPanel
            onNavigateBack={navigateBack}
            onNavigateHome={navigateHome}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;