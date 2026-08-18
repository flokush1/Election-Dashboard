import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, LayersControl, Popup, Tooltip, useMap } from 'react-leaflet';
import { motion } from 'framer-motion';
import { Building, MapPin, Users, BarChart3, Eye, EyeOff } from 'lucide-react';
import { getPartyColor } from '../shared/utils.js';
import { getBoothCoordinates, hasDetailedBoothData, getBoothMetadata } from '../shared/coordinates.js';
import L from 'leaflet';
import { useBoothGeoData } from '../features/maps/BoothDetailMap/useBoothGeoData.js';
import BuildingDetailView from '../features/maps/BoothDetailMap/BuildingDetailView.jsx';

const { BaseLayer, Overlay } = LayersControl;

// Component to auto-fit map bounds to show all data
const AutoFitBounds = ({ boothBoundaryData, buildingData, layerVisibility }) => {
  const map = useMap();

  useEffect(() => {
    const features = [];
    
    if (boothBoundaryData && layerVisibility.boothBoundary) {
      features.push(...boothBoundaryData.features);
    }
    
    if (buildingData && layerVisibility.buildings) {
      features.push(...buildingData.features);
    }

    if (features.length > 0) {
      try {
        const group = new L.FeatureGroup();
        
        features.forEach(feature => {
          if (feature.geometry) {
            const layer = L.geoJSON(feature);
            group.addLayer(layer);
          }
        });

        if (group.getLayers().length > 0) {
          map.fitBounds(group.getBounds(), { padding: [20, 20] });
        }
      } catch (error) {
        console.error('Error fitting bounds:', error);
      }
    }
  }, [map, boothBoundaryData, buildingData, layerVisibility]);

  return null;
};

const BoothDetailMap = ({ 
  boothNumber = "1",
  assemblyConstituency = "NEW DELHI",
  electoralData = null,
  center = null, 
  zoom = 1,
  onBuildingClick = null
}) => {
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [layerVisibility, setLayerVisibility] = useState({
    boothBoundary: true,
    buildings: true
  });

  const mapCenter = center || getBoothCoordinates(assemblyConstituency, boothNumber);
  const boothMetadata = getBoothMetadata(assemblyConstituency, boothNumber);
  const hasDetailedData = hasDetailedBoothData(assemblyConstituency, boothNumber);
  const { boothBoundaryData, buildingData, loading, error } = useBoothGeoData(
    assemblyConstituency,
    boothNumber,
    hasDetailedData
  );


  const getBoothBoundaryStyle = (feature) => {
    // Get party color if electoral data is available
    let fillColor = '#10B981'; // Emerald green
    
    if (electoralData && electoralData.Winner) {
      fillColor = getPartyColor(electoralData.Winner);
    }

    return {
      fillColor: fillColor,
      weight: 5, // Thick border
      opacity: 1,
      color: '#065F46', // Dark green border
      fillOpacity: 0.4, // Semi-transparent
      dashArray: '8, 4' // Dashed border for booth boundary
    };
  };

  const getBuildingStyle = (feature, index) => {
    // Check if this feature is the selected one
    const isSelected = selectedBuilding && 
                      selectedBuilding.properties && 
                      feature.properties &&
                      selectedBuilding.properties.Id === feature.properties.Id;
    
    const props = feature.properties;
    
    // Determine building color based on predicted winner
    let fillColor = '#8B5CF6'; // Default purple for buildings without prediction data
    
    if (props.avg_prob_BJP || props.avg_prob_Congress || props.avg_prob_AAP || props.avg_prob_Others) {
      const parties = {
        'BJP': props.avg_prob_BJP || 0,
        'Congress': props.avg_prob_Congress || 0,
        'AAP': props.avg_prob_AAP || 0,
        'Others': props.avg_prob_Others || 0
      };
      
      // Find the party with highest probability
      const winner = Object.keys(parties).reduce((a, b) => parties[a] > parties[b] ? a : b);
      fillColor = getPartyColor(winner);
    }
    
    return {
      fillColor: isSelected ? '#F97316' : fillColor, // Orange for selected, party color otherwise
      weight: isSelected ? 4 : 3, // Moderate borders
      opacity: 1,
      color: isSelected ? '#9A3412' : '#333', // Darker borders
      fillOpacity: isSelected ? 0.9 : 0.7 // Good opacity
    };
  };

  const onEachBoothFeature = (feature, layer) => {
    const props = feature.properties;
    
    // Add popup with booth information
    const popupContent = `
      <div class="p-2">
        <h3 class="font-bold text-lg mb-2">${props.A_CNST_NM}</h3>
        <div class="space-y-1 text-sm">
          <p><strong>Booth Number:</strong> ${props.Booth_No}</p>
          <p><strong>PC Name:</strong> ${props.PC_Name}</p>
          <p><strong>ED Name:</strong> ${props.ED_Name}</p>
          <p><strong>AC Number:</strong> ${props.AC_No}</p>
          ${electoralData && electoralData.Winner ? 
            `<p><strong>Winning Party:</strong> <span style="color: ${getPartyColor(electoralData.Winner)}">${electoralData.Winner}</span></p>` : 
            ''
          }
        </div>
      </div>
    `;
    
    layer.bindPopup(popupContent);
    
    // Add hover effects
    layer.on({
      mouseover: (e) => {
        layer.setStyle({
          weight: 4,
          fillOpacity: 0.5
        });
      },
      mouseout: (e) => {
        layer.setStyle(getBoothBoundaryStyle(feature));
      }
    });
  };

  const onEachBuildingFeature = (feature, layer) => {
    const props = feature.properties;
    const buildingIndex = buildingData?.features.indexOf(feature) || 0;
    
    // Calculate building area using turf.area for accuracy (handles clipped geometries)
    let area = 0;
    try {
      area = turf.area(feature);
    } catch (areaErr) {
      // Fallback to approximate calculation
      const coords = feature.geometry.coordinates[0];
      area = calculatePolygonArea(coords);
    }
    
    // Format Parcel ID with / instead of |
    const formatParcelId = (parcelId) => {
      if (!parcelId) return null;
      return parcelId.replace(/\|/g, '/');
    };
    
    // Determine display name based on building type
    const getBuildingDisplayName = (props) => {
      const name = props.NAME?.toUpperCase() || '';
      const parcelId = formatParcelId(props.Parcel_No);
      
      // List of special building types that should use NAME instead of Parcel ID
      const specialTypes = [
        'PARK', 'GARDEN', 'PLAYGROUND',
        'SCHOOL', 'COLLEGE', 'UNIVERSITY', 'INSTITUTE',
        'TEMPLE', 'MOSQUE', 'CHURCH', 'GURUDWARA', 'MANDIR', 'MASJID',
        'HOSPITAL', 'CLINIC', 'DISPENSARY',
        'MARKET', 'MALL', 'SHOPPING',
        'STADIUM', 'SPORTS',
        'GOVERNMENT', 'OFFICE', 'MUNICIPAL',
        'COMMUNITY', 'CENTER', 'HALL'
      ];
      
      // Check if name contains any special type
      const isSpecialBuilding = specialTypes.some(type => name.includes(type));
      
      if (isSpecialBuilding && props.NAME) {
        return props.NAME;
      }
      
      // For residential/commercial properties, use Parcel ID
      return parcelId || props.PLOT_NO || 'Building';
    };
    
    // Build simplified tooltip text
    const hasPlotInfo = props.PLOT_NO || props.NAME || props.Road_No;
    const hasVoterData = props.voter_count > 0;
    
    // Use the same logic for tooltip
    const displayName = getBuildingDisplayName(props);
    const title = displayName;
    
    let tooltipText = title;
    if (hasVoterData) {
      tooltipText += ` (${props.voter_count} voters)`;
    }
    
    layer.bindTooltip(tooltipText, { 
      permanent: false, 
      direction: 'center',
      className: 'building-tooltip'
    });
    
    // Add interaction handlers
    layer.on({
      mouseover: (e) => {
        layer.setStyle({
          weight: 4,
          fillOpacity: 0.9
        });
      },
      mouseout: (e) => {
        layer.setStyle(getBuildingStyle(feature, buildingIndex));
      },
      click: () => {
        setSelectedBuilding(feature);
        if (onBuildingClick) {
          onBuildingClick({
            buildingIndex: buildingIndex + 1,
            feature: feature,
            area: area
          });
        }
      }
    });
  };

  // Simple polygon area calculation (Shoelace formula)
  const calculatePolygonArea = (coordinates) => {
    if (!coordinates || coordinates.length < 3) return 0;
    
    let area = 0;
    const n = coordinates.length;
    
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += coordinates[i][0] * coordinates[j][1];
      area -= coordinates[j][0] * coordinates[i][1];
    }
    
    // Convert from degrees to approximate meters (rough calculation)
    area = Math.abs(area) / 2;
    area = area * 111319.9 * 111319.9; // Rough conversion to square meters
    
    return area;
  };

  const toggleLayerVisibility = (layerName) => {
    setLayerVisibility(prev => ({
      ...prev,
      [layerName]: !prev[layerName]
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-gray-600">Loading booth map data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96 bg-red-50 rounded-lg">
        <div className="text-center text-red-600">
          <MapPin className="w-8 h-8 mx-auto mb-2" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

  // Show basic map with center point if no detailed data available
  if (!hasDetailedData) {
    return (
      <motion.div 
        className="bg-white rounded-lg shadow-lg overflow-hidden"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="p-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold flex items-center">
                <MapPin className="w-5 h-5 mr-2" />
                Booth {boothNumber} - {assemblyConstituency}
              </h3>
              <p className="text-blue-100 text-sm">General location map (detailed building data not available)</p>
            </div>
          </div>
        </div>

        {/* Map Container */}
        <div className="h-96 relative">
          <MapContainer
            center={mapCenter}
            zoom={15}
            style={{ height: '100%', width: '100%' }}
            zoomControl={true}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </MapContainer>
        </div>

        {/* Info Panel */}
        <div className="p-4 bg-gray-50 border-t">
          <div className="text-center text-gray-600">
            <Building className="w-8 h-8 mx-auto mb-2" />
            <p className="text-sm">Basic location map for this booth</p>
          </div>
        </div>
      </motion.div>
    );
  }

  // Debug logging
  console.log('🗺️ BoothDetailMap RENDER:');
  console.log('  - Booth:', boothNumber, 'Assembly:', assemblyConstituency);
  console.log('  - hasDetailedData:', hasDetailedData);
  console.log('  - loading:', loading, 'error:', error);
  console.log('  - boothBoundaryData:', boothBoundaryData ? `${boothBoundaryData.features?.length} features` : 'NULL');
  console.log('  - buildingData:', buildingData ? `${buildingData.features?.length} features` : 'NULL');
  console.log('  - layerVisibility:', layerVisibility);
  console.log('  - mapCenter:', mapCenter);

  return (
    <motion.div 
      className="bg-white rounded-lg shadow-lg overflow-hidden"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold flex items-center">
              <MapPin className="w-5 h-5 mr-2" />
              Booth {boothNumber} - {assemblyConstituency}
            </h3>
            {boothMetadata ? (
              <>
                <p className="text-blue-100 text-sm">{boothMetadata.name}</p>
                <p className="text-blue-200 text-xs">({boothMetadata.address})</p>
                <p className="text-blue-200 text-xs mt-1">
                  Assembly: {assemblyConstituency} | Ward: {boothMetadata.ward} | Locality: {boothMetadata.locality}
                </p>
              </>
            ) : (
              <p className="text-blue-100 text-sm">Detailed booth boundary and building map</p>
            )}
          </div>
          
          {/* Layer Controls */}
          <div className="flex space-x-2">
            <button
              onClick={() => toggleLayerVisibility('boothBoundary')}
              className={`flex items-center px-3 py-1 rounded text-xs transition-colors ${
                layerVisibility.boothBoundary 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-white text-blue-600'
              }`}
            >
              {layerVisibility.boothBoundary ? <Eye className="w-3 h-3 mr-1" /> : <EyeOff className="w-3 h-3 mr-1" />}
              Boundary
            </button>
            <button
              onClick={() => toggleLayerVisibility('buildings')}
              className={`flex items-center px-3 py-1 rounded text-xs transition-colors ${
                layerVisibility.buildings 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-white text-blue-600'
              }`}
            >
              {layerVisibility.buildings ? <Eye className="w-3 h-3 mr-1" /> : <EyeOff className="w-3 h-3 mr-1" />}
              Buildings
            </button>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="h-[600px] relative">
        <MapContainer
          center={mapCenter}
          zoom={zoom}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {/* Auto-fit bounds to show all data */}
          <AutoFitBounds 
            boothBoundaryData={boothBoundaryData}
            buildingData={buildingData}
            layerVisibility={layerVisibility}
          />
          
          {/* Booth Boundary Layer */}
          {boothBoundaryData && layerVisibility.boothBoundary && (
            <GeoJSON
              key="booth-boundary"
              data={boothBoundaryData}
              style={getBoothBoundaryStyle}
              onEachFeature={onEachBoothFeature}
              onAdd={() => console.log('Booth boundary layer added to map')}
            />
          )}
          
          {/* Buildings Layer */}
          {buildingData && layerVisibility.buildings && (
            <GeoJSON
              key="buildings"
              data={buildingData}
              style={(feature) => {
                const index = buildingData.features.indexOf(feature);
                return getBuildingStyle(feature, index);
              }}
              onEachFeature={onEachBuildingFeature}
            />
          )}
        </MapContainer>
      </div>

      {/* Info Panel */}
      <div className="p-4 bg-gray-50 border-t">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center mb-4">
          <div className="bg-white p-3 rounded-lg shadow-sm">
            <div className={`text-lg font-bold ${layerVisibility.boothBoundary ? 'text-blue-600' : 'text-gray-400'}`}>
              {boothBoundaryData?.features?.length || 0}
            </div>
            <div className="text-xs text-gray-600">
              Booth Boundaries {layerVisibility.boothBoundary ? '(Visible)' : '(Hidden)'}
            </div>
          </div>
          <div className="bg-white p-3 rounded-lg shadow-sm">
            <div className={`text-lg font-bold ${layerVisibility.buildings ? 'text-green-600' : 'text-gray-400'}`}>
              {buildingData?.features?.length || 0}
            </div>
            <div className="text-xs text-gray-600">
              Buildings {layerVisibility.buildings ? '(Visible)' : '(Hidden)'}
            </div>
          </div>
          <div className="bg-white p-3 rounded-lg shadow-sm">
            <div className="text-lg font-bold text-purple-600">
              {selectedBuilding ? (
                selectedBuilding.properties?.PLOT_NO || 
                selectedBuilding.properties?.NAME || 
                'Selected'
              ) : '-'}
            </div>
            <div className="text-xs text-gray-600">Selected Building</div>
          </div>
          <div className="bg-white p-3 rounded-lg shadow-sm">
            <div className="text-lg font-bold text-orange-600">
              {electoralData?.Winner || 'N/A'}
            </div>
            <div className="text-xs text-gray-600">Winning Party</div>
          </div>
        </div>
        
        {/* Data Status */}
        <div className="text-center text-sm text-gray-600 mb-2">
          Status: 
          {boothBoundaryData ? ' ✅ Boundary Data Loaded' : ' ❌ No Boundary Data'}
          {buildingData ? ' ✅ Building Data Loaded' : ' ❌ No Building Data'}
        </div>
      </div>

      {/* Legend */}
      <div className="px-4 pb-4">
        <div className="flex items-center justify-center space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-4 h-4 border-2 border-green-800 border-dashed bg-green-200 mr-2 rounded"></div>
            <span>Booth Boundary</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-purple-500 border border-purple-700 mr-2 rounded"></div>
            <span>Buildings</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-orange-500 border border-orange-700 mr-2 rounded"></div>
            <span>Selected Building</span>
          </div>
        </div>
      </div>

      {/* Custom CSS for tooltips */}
      <style jsx>{`
        .building-tooltip {
          background: rgba(0, 0, 0, 0.8) !important;
          color: white !important;
          border: none !important;
          border-radius: 4px !important;
          font-size: 12px !important;
          padding: 4px 8px !important;
        }
      `}</style>
      <BuildingDetailView building={selectedBuilding} onClose={() => setSelectedBuilding(null)} />
    </motion.div>
  );
};

export default BoothDetailMap;