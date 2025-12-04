import React from 'react';
import { MapContainer, TileLayer, GeoJSON, Popup, Tooltip } from 'react-leaflet';
import { motion } from 'framer-motion';
import { getPartyColor, canonicalWardKey, normalizeWardDisplay } from '../shared/utils.js';

const InteractiveMap = ({ 
  geoData, 
  electoralData, 
  center = [28.6139, 77.2090], 
  zoom = 11,
  onItemClick,
  level = 'assembly',
  selectedItem = null
}) => {
  const getFeatureName = (feature, level) => {
    const props = feature.properties;
    switch(level) {
      case 'parliament': return props.PC_Name;
      case 'assembly': return props.A_CNST_NM;
      case 'ward': return props.WardName;
      case 'booth': return props.BoothId || props.PartNo || props.name || 'Unknown';
      default: return props.name || 'Unknown';
    }
  };

  // Helper: find electoral info by robustly matching names (direct, display-normalized, canonical key)
  const findElectoralInfo = (name, dataObj) => {
    if (!name || !dataObj) return { key: null, info: null };
    // 1) direct
    if (dataObj[name]) return { key: name, info: dataObj[name] };
    // 2) title/display normalized
    const disp = normalizeWardDisplay(name);
    if (dataObj[disp]) return { key: disp, info: dataObj[disp] };
    // 3) canonical key match
    const targetKey = canonicalWardKey(name);
    let matchKey = null;
    for (const k of Object.keys(dataObj)) {
      if (canonicalWardKey(k) === targetKey) { matchKey = k; break; }
    }
    return matchKey ? { key: matchKey, info: dataObj[matchKey] } : { key: null, info: null };
  };

  // Debug logging - log once when component mounts
  if (geoData && geoData.features && electoralData) {
    console.log('InteractiveMap debug info:', {
      geoDataFeatures: geoData.features.length,
      electoralDataKeys: Object.keys(electoralData).slice(0, 5),
      level,
      sampleGeoNames: geoData.features.slice(0, 3).map(f => getFeatureName(f, level)),
      sampleElectoralKeys: Object.keys(electoralData).slice(0, 3)
    });
  }
  const getFeatureStyle = (feature) => {
    const featureName = getFeatureName(feature, level);
    
    // Get winning party for this feature
  let winningParty = null;
  let fillColor = '#9CA3AF'; // Default neutral gray when no match (avoid implying AAP)
    
    if (electoralData && featureName) {
      // Try to find electoral data for this feature
      let electoralInfo = null;
      let matchedKey = null;
      
      if (level === 'assembly') {
        const res = findElectoralInfo(featureName, electoralData);
        matchedKey = res.key; electoralInfo = res.info;
      } else if (level === 'booth') {
        // For booth level, electoralData is an array of booth objects
        if (Array.isArray(electoralData)) {
          // Try to match by PartNo, BoothId, or name
          electoralInfo = electoralData.find(booth => {
            return booth.PartNo?.toString() === featureName ||
                   booth.BoothId?.toString() === featureName ||
                   booth.name === featureName ||
                   `Booth ${booth.PartNo}` === featureName;
          });
        }
      } else if (level === 'ward') {
        const res = findElectoralInfo(featureName, electoralData);
        matchedKey = res.key; electoralInfo = res.info;
        console.log(`🏘️ Ward lookup: "${featureName}" -> "${matchedKey || '(no match)'}"`);
        if (!electoralInfo) {
          console.log(`✗ No match. Sample keys:`, Object.keys(electoralData).slice(0, 8));
        }
      } else {
        // For other levels, try direct lookup
        electoralInfo = electoralData[featureName];
      }
      
      // If we found electoral data, determine winning party
      if (electoralInfo) {
        if (level === 'booth' && electoralInfo.Winner) {
          // For booths, use the Winner field directly
          winningParty = electoralInfo.Winner;
          fillColor = getPartyColor(winningParty);
        } else if (level === 'assembly' || level === 'ward') {
          // For assembly and ward levels, use boothsWon to determine the winning party
          // This shows which party won the most booths in that area
          if (electoralInfo.boothsWon) {
            const boothEntries = Object.entries(electoralInfo.boothsWon)
              .filter(([party]) => party !== 'Tie'); // Exclude Tie from consideration
            if (boothEntries.length > 0) {
              winningParty = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
              fillColor = getPartyColor(winningParty);
            }
          }
        } else if (electoralInfo.partyVotes) {
          // For other levels (like parliament), calculate from partyVotes
          const partyEntries = Object.entries(electoralInfo.partyVotes);
          if (partyEntries.length > 0) {
            winningParty = partyEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            fillColor = getPartyColor(winningParty);
          }
        }
      }
    }
    
    return {
      fillColor: fillColor,
      weight: selectedItem === featureName ? 3 : 2,
      opacity: 1,
      color: selectedItem === featureName ? '#1F2937' : '#ffffff',
      fillOpacity: selectedItem === featureName ? 0.8 : 0.6
    };
  };  const onEachFeature = (feature, layer) => {
    const featureName = getFeatureName(feature, level);
    
    layer.on({
      mouseover: (e) => {
        const layer = e.target;
        layer.setStyle({
          weight: 3,
          color: '#1F2937',
          fillOpacity: 0.8
        });
      },
      mouseout: (e) => {
        const layer = e.target;
        layer.setStyle(getFeatureStyle(feature));
      },
      click: (e) => {
        console.log('Map clicked - Feature Name:', featureName);
        console.log('Map Level:', level);
        console.log('onItemClick function:', typeof onItemClick);
        
        if (onItemClick) {
          let nameForNavigation = featureName;
          
          if (level === 'assembly' || level === 'ward') {
            // Resolve to the exact key used in electoral data if possible
            const res = findElectoralInfo(featureName, electoralData);
            nameForNavigation = res.key || normalizeWardDisplay(featureName);
          } else if (level === 'booth') {
            // For booth level, use the feature name as-is (could be PartNo or BoothId)
            nameForNavigation = featureName;
          }
          
          console.log('Original name:', featureName);
          console.log('Converted name for navigation:', nameForNavigation);
          onItemClick(nameForNavigation);
        } else {
          console.log('No onItemClick handler provided');
        }
      }
    });

    // Enhanced popup with electoral information
    let electoralInfo = null;
    let winningParty = null;
    let totalVotes = 0;
    let additionalInfo = '';
    
    if (electoralData && featureName) {
      if (level === 'assembly') {
        electoralInfo = findElectoralInfo(featureName, electoralData).info;
        
        if (electoralInfo) {
          // For assembly level, determine winning party by most booths won
          if (electoralInfo.boothsWon) {
            const boothEntries = Object.entries(electoralInfo.boothsWon)
              .filter(([party]) => party !== 'Tie');
            if (boothEntries.length > 0) {
              winningParty = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }
          }
          // Use partyVotes for total votes display
          if (electoralInfo.partyVotes) {
            const partyEntries = Object.entries(electoralInfo.partyVotes);
            totalVotes = partyEntries.reduce((sum, [, votes]) => sum + votes, 0);
          }
        }
      } else if (level === 'booth') {
        if (Array.isArray(electoralData)) {
          electoralInfo = electoralData.find(booth => {
            return booth.PartNo?.toString() === featureName ||
                   booth.BoothId?.toString() === featureName ||
                   booth.name === featureName ||
                   `Booth ${booth.PartNo}` === featureName;
          });
          
          if (electoralInfo) {
            winningParty = electoralInfo.Winner;
            totalVotes = electoralInfo.Total_Polled || 0;
            additionalInfo = `
              <div class="text-xs text-gray-600 space-y-1">
                <div>Population: ${(electoralInfo.TotalPop || 0).toLocaleString()}</div>
                <div>Margin: ${Math.round(electoralInfo.Margin || 0)}</div>
                ${electoralInfo.Address ? `<div>Address: ${electoralInfo.Address}</div>` : ''}
              </div>
            `;
          }
        }
      } else if (level === 'ward') {
        electoralInfo = findElectoralInfo(featureName, electoralData).info;
        
        if (electoralInfo) {
          // For ward level, determine winning party by most booths won
          if (electoralInfo.boothsWon) {
            const boothEntries = Object.entries(electoralInfo.boothsWon)
              .filter(([party]) => party !== 'Tie');
            if (boothEntries.length > 0) {
              winningParty = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }
          }
          // Use partyVotes for total votes display
          if (electoralInfo.partyVotes) {
            const partyEntries = Object.entries(electoralInfo.partyVotes);
            totalVotes = partyEntries.reduce((sum, [, votes]) => sum + votes, 0);
          }
        }
      } else {
        electoralInfo = electoralData[featureName];
        if (electoralInfo && electoralInfo.partyVotes) {
          const partyEntries = Object.entries(electoralInfo.partyVotes);
          if (partyEntries.length > 0) {
            winningParty = partyEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            totalVotes = partyEntries.reduce((sum, [, votes]) => sum + votes, 0);
          }
        }
      }
    }
    
    const popupContent = `
      <div class="p-3 min-w-48">
        <h3 class="font-bold text-lg mb-2">${featureName}</h3>
        ${winningParty ? `
          <div class="mb-2">
            <div class="flex items-center gap-2 mb-1">
              <div class="w-3 h-3 rounded-full" style="background-color: ${getPartyColor(winningParty)}"></div>
              <span class="font-semibold text-sm">Leading: ${winningParty}</span>
            </div>
            <div class="text-xs text-gray-600">
              Total Votes: ${totalVotes.toLocaleString()}
            </div>
            ${additionalInfo}
          </div>
        ` : ''}
        <div class="text-sm text-gray-600">
          Click to view detailed analysis
        </div>
      </div>
    `;
    
    layer.bindPopup(popupContent);
    
    // Add permanent tooltip to show assembly name
    layer.bindTooltip(featureName, {
      permanent: true,
      direction: 'center',
      className: 'assembly-label',
      opacity: 0.9
    });
  };

  if (!geoData || !geoData.features) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-gray-600">Loading map data...</p>
          <p className="text-xs text-gray-500 mt-2">
            GeoData: {geoData ? 'Present' : 'Missing'} | 
            Features: {geoData?.features ? geoData.features.length : 'None'}
          </p>
        </div>
      </div>
    );
  }

  // Create dynamic legend based on level and electoral data
  const MapLegend = () => {
    const legendData = {
      parliament: { title: 'Parliamentary Constituency', color: '#3B82F6' },
      assembly: { title: 'Assembly Constituencies', color: '#3B82F6' },
      ward: { title: 'Ward Boundaries', color: '#3B82F6' },
      booth: { title: 'Polling Booths', color: '#3B82F6' }
    };
    
    const currentLegend = legendData[level] || legendData.assembly;
    
    // Get unique parties from electoral data for legend
    const parties = new Set();
    if (electoralData) {
      if (level === 'booth' && Array.isArray(electoralData)) {
        // For booth level, extract from Winner field
        electoralData.forEach(booth => {
          if (booth && booth.Winner && booth.Winner !== 'Unknown') {
            parties.add(booth.Winner);
          }
        });
      } else {
        // For assembly/ward, prefer boothsWon; otherwise fallback to partyVotes
        Object.values(electoralData).forEach(area => {
          if (!area) return;
          let winner = null;
          if (area.boothsWon) {
            const boothEntries = Object.entries(area.boothsWon).filter(([p]) => p !== 'Tie');
            if (boothEntries.length > 0) {
              winner = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }
          }
          if (!winner && area.partyVotes) {
            const partyEntries = Object.entries(area.partyVotes);
            if (partyEntries.length > 0) {
              winner = partyEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }
          }
          if (winner) parties.add(winner);
        });
      }
    }
    
    return (
      <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 z-10 border">
        <h4 className="text-sm font-semibold mb-2">Map Legend</h4>
        <div className="space-y-1">
          {parties.size > 0 ? (
            <>
              <div className="text-xs text-gray-600 mb-2">Winning Party by Area:</div>
              {Array.from(parties).map(party => (
                <div key={party} className="flex items-center space-x-2">
                  <div 
                    className="w-4 h-4 rounded border"
                    style={{ backgroundColor: getPartyColor(party) }}
                  ></div>
                  <span className="text-xs text-gray-700">{party}</span>
                </div>
              ))}
            </>
          ) : (
            <div className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 rounded border"
                style={{ backgroundColor: currentLegend.color }}
              ></div>
              <span className="text-xs text-gray-700">{currentLegend.title}</span>
            </div>
          )}
          <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
            Click to view detailed analysis
          </div>
        </div>
      </div>
    );
  };

  return (
    <motion.div 
      className="h-full w-full rounded-lg overflow-hidden shadow-lg relative"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <GeoJSON
          data={geoData}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
          key={`${level}-${selectedItem}`}
        />
      </MapContainer>
      <MapLegend />
    </motion.div>
  );
};

export default InteractiveMap;