import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, LayersControl, Popup, Tooltip, useMap } from 'react-leaflet';
import { motion } from 'framer-motion';
import { Building, MapPin, Users, BarChart3, Eye, EyeOff } from 'lucide-react';
import { getPartyColor } from '../shared/utils.js';
import { getBoothCoordinates, hasDetailedBoothData, getBoothMetadata } from '../shared/coordinates.js';
import L from 'leaflet';
// turf is used for spatial clipping of plot polygons by booth boundary
import * as turf from '@turf/turf';

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
  zoom = 17,
  onBuildingClick = null
}) => {
  const [boothBoundaryData, setBoothBoundaryData] = useState(null);
  const [buildingData, setBuildingData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [layerVisibility, setLayerVisibility] = useState({
    boothBoundary: true,
    buildings: true
  });

  // Use provided center or get from coordinate mapping
  const mapCenter = center || getBoothCoordinates(assemblyConstituency, boothNumber);
  
  // Get booth metadata for display
  const boothMetadata = getBoothMetadata(assemblyConstituency, boothNumber);
  const hasDetailedData = hasDetailedBoothData(assemblyConstituency, boothNumber);

  useEffect(() => {
    const loadGeoData = async () => {
      try {
        setLoading(true);
        console.log('BoothDetailMap: Loading data for booth', boothNumber, 'in', assemblyConstituency);
        console.log('BoothDetailMap: assemblyConstituency value:', assemblyConstituency);
        console.log('BoothDetailMap: assemblyConstituency type:', typeof assemblyConstituency);
        console.log('Has detailed data:', hasDetailedData);
        
        // Only try to load detailed geospatial data if available
        if (hasDetailedData) {
          console.log('Attempting to load geospatial data...');
          
          // Determine data sources based on assembly/booth
          const normalizedAssembly = assemblyConstituency?.toUpperCase().trim();
          const boothStr = boothNumber?.toString();
          let boothBoundaryUrl = '/data/geospatial/New_Delhi_Booth_Data.geojson';
          let buildingsUrl = '/data/geospatial/New_Delhi_Booth_Buildings.geojson';
          let useClipWithPlots = false;

          // RK Puram Booth 17 uses dedicated files with voter predictions and plot clipping
          if (normalizedAssembly === 'R K PURAM' && boothStr === '17') {
            boothBoundaryUrl = '/data/geospatial/RKPuram_Booth_17_Boundary.geojson';
            buildingsUrl = '/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson';
            useClipWithPlots = true;
          }

          // Load booth boundary data
          try {
            const boothResponse = await fetch(boothBoundaryUrl);
            console.log('Booth data response status:', boothResponse.status);
            if (boothResponse.ok) {
              const boothGeoData = await boothResponse.json();
              console.log('Loaded booth data:', boothGeoData);
              console.log('Total booth features:', boothGeoData.features?.length);
              
              // Filter for the specific booth with case-insensitive assembly matching (if file contains multiple)
              let filteredBooth = boothGeoData;
              if (!useClipWithPlots) {
                filteredBooth = {
                  ...boothGeoData,
                  features: boothGeoData.features.filter(feature => {
                    const boothMatch = feature.properties.Booth_No?.toString() === boothNumber?.toString();
                    const assemblyMatch = feature.properties.A_CNST_NM?.toUpperCase().trim() === assemblyConstituency?.toUpperCase().trim();
                    return boothMatch && assemblyMatch;
                  })
                };
              }
              console.log('Filtered booth data:', filteredBooth);
              console.log('Filtered booth features:', filteredBooth.features?.length);
              setBoothBoundaryData(filteredBooth);
            } else {
              console.error('Failed to load booth data:', boothResponse.status);
            }
          } catch (boothError) {
            console.error('Error loading booth data:', boothError);
          }

          // Load building/plot data
          try {
            const buildingResponse = await fetch(buildingsUrl);
            console.log('Building data response status:', buildingResponse.status);
            if (buildingResponse.ok) {
              const buildingGeoData = await buildingResponse.json();
              console.log('Loaded building data:', buildingGeoData);
              console.log('Total building features:', buildingGeoData.features?.length);
              
              if (useClipWithPlots) {
                // Clip plots to the booth boundary
                try {
                  const boothFeat = (boothBoundaryData?.features && boothBoundaryData.features[0]) ? boothBoundaryData.features[0] : null;
                  // If booth boundary not yet in state (race), create a turf polygon from freshly fetched data
                  const boothForClip = boothFeat || (buildingGeoData && buildingGeoData.type ? null : null);
                  const boundaryCollection = (boothBoundaryData && boothBoundaryData.features && boothBoundaryData.features.length > 0)
                    ? boothBoundaryData
                    : await (async () => {
                        // fetch boundary again if needed
                        const resp = await fetch('/data/geospatial/RKPuram_Booth_17_Boundary.geojson');
                        return await resp.json();
                      })();

                  const boundaryGeom = boundaryCollection.features[0].geometry;
                  
                  console.log('Boundary geometry:', boundaryGeom);
                  console.log('Boundary type:', boundaryGeom.type);
                  console.log('Total plots to clip:', buildingGeoData.features.length);
                  console.log('First plot geometry:', buildingGeoData.features[0]?.geometry);
                  
                  let boundaryPoly;
                  try {
                    if (boundaryGeom.type === 'MultiPolygon') {
                      // Convert MultiPolygon to single Polygon using first polygon
                      // MultiPolygon coordinates are [[[lon, lat], ...]]
                      // Polygon coordinates are [[lon, lat], ...]
                      boundaryPoly = turf.multiPolygon(boundaryGeom.coordinates);
                      console.log('Created MultiPolygon from boundary');
                    } else {
                      boundaryPoly = turf.polygon(boundaryGeom.coordinates);
                      console.log('Created Polygon from boundary');
                    }
                  } catch (polyErr) {
                    console.error('Error creating boundary polygon:', polyErr);
                    throw polyErr;
                  }
                  
                  const clippedFeatures = [];
                  let intersectCount = 0;
                  let errorCount = 0;
                  
                  for (let i = 0; i < buildingGeoData.features.length; i++) {
                    const feat = buildingGeoData.features[i];
                    try {
                      const geom = feat.geometry;
                      if (!geom) {
                        console.warn(`Feature ${i} has no geometry`);
                        continue;
                      }
                      
                      if (geom.type !== 'Polygon') {
                        console.warn(`Feature ${i} is not a Polygon, type: ${geom.type}`);
                        continue;
                      }
                      
                      const plotPoly = turf.polygon(geom.coordinates);
                      
                      if (turf.booleanIntersects(plotPoly, boundaryPoly)) {
                        intersectCount++;
                        const clipped = turf.intersect(plotPoly, boundaryPoly);
                        if (clipped) {
                          clippedFeatures.push({ 
                            type: 'Feature', 
                            properties: feat.properties || {}, 
                            geometry: clipped.geometry 
                          });
                        } else {
                          // If intersection returns null, use original plot
                          clippedFeatures.push(feat);
                        }
                      }
                    } catch (clipErr) {
                      errorCount++;
                      if (errorCount <= 5) {
                        console.warn(`Clip error for feature ${i}:`, clipErr.message);
                      }
                    }
                  }

                  console.log('✓ Features that intersect boundary:', intersectCount);
                  console.log('✓ Clipped building features:', clippedFeatures.length);
                  console.log('✗ Clipping errors:', errorCount);
                  if (clippedFeatures.length > 0) {
                    console.log('Sample clipped feature:', clippedFeatures[0]);
                  }
                  
                  const clippedCollection = { type: 'FeatureCollection', features: clippedFeatures };
                  
                  // TEMPORARY: If no features after clipping, show all features for debugging
                  if (clippedFeatures.length === 0) {
                    console.warn('⚠️ NO FEATURES AFTER CLIPPING - Showing all features for debugging');
                    setBuildingData(buildingGeoData);
                  } else {
                    setBuildingData(clippedCollection);
                  }
                } catch (clipError) {
                  console.error('Error clipping plots to boundary:', clipError);
                  // Fallback: render all plots without clipping
                  setBuildingData(buildingGeoData);
                }
              } else {
                // Filter for the specific booth with case-insensitive assembly matching (New Delhi case)
                const filteredBuildings = {
                  ...buildingGeoData,
                  features: buildingGeoData.features.filter(feature => {
                    const boothMatch = feature.properties.Booth_No?.toString() === boothNumber?.toString();
                    const assemblyMatch = feature.properties.A_CNST_NM?.toUpperCase().trim() === assemblyConstituency?.toUpperCase().trim();
                    return boothMatch && assemblyMatch;
                  })
                };
                console.log('Filtered building data:', filteredBuildings);
                console.log('Filtered building features:', filteredBuildings.features?.length);
                setBuildingData(filteredBuildings);
              }
            } else {
              console.error('Failed to load building data:', buildingResponse.status);
            }
          } catch (buildingError) {
            console.error('Error loading building data:', buildingError);
          }
        } else {
          console.log('No detailed data available for this booth');
        }

      } catch (err) {
        console.error('Error loading geospatial data:', err);
        setError('Failed to load map data');
      } finally {
        setLoading(false);
      }
    };

    loadGeoData();
  }, [boothNumber, assemblyConstituency, hasDetailedData]);

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
    
    // Function to open building details in new tab
    const openBuildingInNewTab = (feature, area) => {
      const props = feature.properties;
      const parcelId = formatParcelId(props.Parcel_No);
      const displayName = getBuildingDisplayName(props);
      const isSpecialBuilding = displayName !== parcelId && displayName !== props.PLOT_NO && displayName !== 'Building';
      
      // Get party colors
      const getPartyColor = (party) => {
        const colors = {
          'BJP': '#FF9933',
          'Congress': '#19AAED',
          'AAP': '#0072B0',
          'Others': '#808080',
          'NOTA': '#000000'
        };
        return colors[party] || '#808080';
      };
      
      // Find winner
      let winner = 'Unknown';
      let winnerProb = 0;
      if (props.avg_prob_BJP) {
        const parties = {
          'BJP': props.avg_prob_BJP || 0,
          'Congress': props.avg_prob_Congress || 0,
          'AAP': props.avg_prob_AAP || 0,
          'Others': props.avg_prob_Others || 0
        };
        winner = Object.keys(parties).reduce((a, b) => parties[a] > parties[b] ? a : b);
        winnerProb = parties[winner];
      }
      
      // Generate HTML for the new tab
      const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${displayName} - Electoral Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
            padding: 20px;
            margin: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }
        
        .header h1 {
            font-size: 1.8em;
            margin-bottom: 8px;
        }
        
        .header .subtitle {
            opacity: 0.9;
            font-size: 1em;
        }
        
        .content { padding: 30px; }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .info-item {
            display: flex;
            flex-direction: column;
        }
        
        .info-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #333;
        }
        
        .prediction-summary {
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .prediction-summary h3 {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.2em;
        }
        
        .party-bars {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .party-bar {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .party-name {
            min-width: 80px;
            font-weight: 600;
        }
        
        .party-bar-bg {
            flex: 1;
            height: 28px;
            background: #e0e0e0;
            border-radius: 14px;
            overflow: hidden;
            position: relative;
        }
        
        .party-bar-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: 600;
            font-size: 0.9em;
            transition: width 0.5s ease;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #333;
            font-size: 1.4em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
            animation: slideIn 1s ease;
        }
        
        .party-item:hover {
            transform: translateX(10px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        
        .party-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .party-name {
            font-size: 1.3em;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .party-percentage {
            font-size: 2em;
            font-weight: bold;
        }
        
        .party-bar-container {
            height: 40px;
            background: #e0e0e0;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .party-bar-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 0 15px;
            color: white;
            font-weight: 600;
            font-size: 1.1em;
            transition: width 1s ease-out;
            position: relative;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .party-bar-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .party-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        
        .party-stat {
            text-align: center;
            padding: 10px;
            background: #f7f7f7;
            border-radius: 8px;
        }
        
        .party-stat-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .party-stat-label {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        
        .turnout-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #90caf9 100%);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .turnout-box .turnout-label {
            font-size: 1.2em;
            color: #1976d2;
            margin-bottom: 10px;
        }
        
        .turnout-box .turnout-value {
            font-size: 3em;
            font-weight: bold;
            color: #0d47a1;
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .detail-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        
        .detail-card:hover {
            transform: translateY(-3px);
        }
        
        .detail-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .detail-value {
            font-size: 1.5em;
            font-weight: 700;
            color: #333;
        }
        
        .voter-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .voter-table thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .voter-table th {
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .voter-table tbody tr {
            background: white;
            transition: all 0.3s ease;
        }
        
        .voter-table tbody tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        .voter-table tbody tr:hover {
            background: #e3f2fd;
        }
        
        .voter-table td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .voter-row {
            cursor: pointer;
        }
        
        .voter-row.expanded {
            background: #e8f5e9 !important;
        }
        
        .voter-details-row {
            display: none;
        }
        
        .voter-details-row.show {
            display: table-row;
            background: #f1f8f4 !important;
        }
        
        .voter-details-cell {
            padding: 20px !important;
            border-bottom: 2px solid #667eea !important;
        }
        
        .voter-details-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .detail-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .detail-box h4 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        
        .detail-box .value {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }
        
        .prob-bars {
            margin-top: 10px;
        }
        
        .prob-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .prob-label {
            min-width: 70px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .prob-bar-bg {
            flex: 1;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            display: flex;
            align-items: center;
            padding: 0 8px;
            color: white;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .expand-icon {
            color: #667eea;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .voter-party-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            color: white;
        }
        
        @media print {
            body { background: white; padding: 0; }
            .container { box-shadow: none; }
            .party-item:hover, .voter-table tr:hover { transform: none; }
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 1.8em; }
            .stat-card { padding: 15px; }
            .party-percentage { font-size: 1.5em; }
            .voter-table { font-size: 0.9em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <h1>${isSpecialBuilding ? '🏢 ' + displayName : '🏠 Building Details'}</h1>
                <div class="subtitle">${isSpecialBuilding ? (parcelId ? `Parcel: ${parcelId}` : '') : displayName}</div>
                <div class="header-badges">
                    ${props.voter_count ? `<div class="badge">👥 ${props.voter_count} Registered Voters</div>` : ''}
                    ${winner !== 'Unknown' ? `<div class="badge winner">🏆 ${winner} Predicted to Win</div>` : ''}
                    ${props.avg_turnout_prob ? `<div class="badge">📊 ${(props.avg_turnout_prob * 100).toFixed(1)}% Expected Turnout</div>` : ''}
                </div>
            </div>
        </div>
        
        <div class="content">
            <!-- Building Info -->
            <div class="info-grid">
                ${props.Parcel_No ? `
                <div class="info-item">
                    <div class="info-label">Parcel ID</div>
                    <div class="info-value">${parcelId}</div>
                </div>` : ''}
                ${props.PLOT_NO ? `
                <div class="info-item">
                    <div class="info-label">Plot Number</div>
                    <div class="info-value">${props.PLOT_NO}</div>
                </div>` : ''}
                ${props.Road_No ? `
                <div class="info-item">
                    <div class="info-label">Road Number</div>
                    <div class="info-value">${props.Road_No}</div>
                </div>` : ''}
                ${props.voter_count ? `
                <div class="info-item">
                    <div class="info-label">Registered Voters</div>
                    <div class="info-value">${props.voter_count}</div>
                </div>` : ''}
                <div class="info-item">
                    <div class="info-label">Area</div>
                    <div class="info-value">${area.toFixed(0)} sq m</div>
                </div>
                ${props.avg_turnout_prob ? `
                <div class="info-item">
                    <div class="info-label">Expected Turnout</div>
                    <div class="info-value">${(props.avg_turnout_prob * 100).toFixed(1)}%</div>
                </div>` : ''}
            </div>
            
            <!-- Voters List with Expandable Details -->
            ${props.voters && props.voters.length > 0 ? `
            <div class="section">
                <h2>👥 Voter Details (${props.voters.length}) - Click to Expand</h2>
                <table class="voter-table">
                    <thead>
                        <tr>
                            <th style="width: 40px"></th>
                            <th>#</th>
                            <th>Name</th>
                            <th>Age</th>
                            <th>Gender</th>
                            <th>Predicted Party</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${props.voters.map((voter, idx) => `
                        <tr class="voter-row" onclick="toggleVoterDetails(${idx})">
                            <td><span class="expand-icon" id="icon-${idx}">▶</span></td>
                            <td><strong>${idx + 1}</strong></td>
                            <td><strong>${voter.name}</strong></td>
                            <td>${voter.age || 'N/A'}</td>
                            <td>${voter.gender === 'MALE' ? '♂️ M' : voter.gender === 'FEMALE' ? '♀️ F' : 'N/A'}</td>
                            <td>
                                ${voter.predicted_party ? `
                                    <span class="voter-party-badge" style="background: ${getPartyColor(voter.predicted_party)}">
                                        ${voter.predicted_party}
                                    </span>
                                ` : 'N/A'}
                            </td>
                            <td><strong>${voter.predicted_party && voter['prob_' + voter.predicted_party] ? (voter['prob_' + voter.predicted_party] * 100).toFixed(1) + '%' : 'N/A'}</strong></td>
                        </tr>
                        <tr class="voter-details-row" id="details-${idx}">
                            <td colspan="7" class="voter-details-cell">
                                <div class="voter-details-content">
                                    ${(() => {
                                        // Calculate alignment
                                        const probs = [
                                            voter.prob_BJP || 0,
                                            voter.prob_Congress || 0,
                                            voter.prob_AAP || 0,
                                            voter.prob_Others || 0
                                        ];
                                        const maxProb = Math.max(...probs);
                                        let alignment = 'Swing Voter';
                                        let alignColor = '#dc2626';
                                        let alignBg = '#fee2e2';
                                        let alignDesc = 'This voter has not demonstrated strong party loyalty and could be persuaded by any party.';
                                        
                                        if (maxProb >= 0.7) {
                                            alignment = 'Core Supporter';
                                            alignColor = '#16a34a';
                                            alignBg = '#dcfce7';
                                            alignDesc = 'This voter shows strong party loyalty and is unlikely to change their preference.';
                                        } else if (maxProb >= 0.4) {
                                            alignment = 'Leaning Voter';
                                            alignColor = '#f59e0b';
                                            alignBg = '#fef3c7';
                                            alignDesc = 'This voter shows a preference but may still be open to persuasion.';
                                        }
                                        
                                        return `
                                        <div class="detail-box" style="background: ${alignBg}; border-left: 4px solid ${alignColor};">
                                            <h4 style="color: ${alignColor}">🎯 Voter Alignment</h4>
                                            <div class="value" style="color: ${alignColor}; font-size: 1.3em; margin-bottom: 8px;">
                                                ${alignment}
                                            </div>
                                            <div style="background: rgba(255,255,255,0.7); padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                                    <span style="font-weight: 600; color: #333;">Commitment Level</span>
                                                    <span style="font-weight: 700; color: ${alignColor}; font-size: 1.1em;">${(maxProb * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                            <p style="font-size: 0.9em; color: #555; line-height: 1.5; margin: 0;">
                                                ${alignDesc}
                                            </p>
                                        </div>`;
                                    })()}
                                    ${voter.turnout_prob ? `
                                    <div class="detail-box">
                                        <h4>📊 Turnout Probability</h4>
                                        <div class="value">${(voter.turnout_prob * 100).toFixed(1)}%</div>
                                        <div class="prob-bars">
                                            <div class="prob-bar-bg">
                                                <div class="prob-bar-fill" style="width: ${(voter.turnout_prob * 100).toFixed(1)}%; background: #2196f3;">
                                                    ${voter.turnout_prob > 0.5 ? 'Likely to Vote' : 'May Not Vote'}
                                                </div>
                                            </div>
                                        </div>
                                    </div>` : ''}
                                    <div class="detail-box">
                                        <h4>🗳️ Party Preferences</h4>
                                        <div class="prob-bars">
                                            ${voter.prob_BJP ? `
                                            <div class="prob-item">
                                                <div class="prob-label" style="color: ${getPartyColor('BJP')}">BJP</div>
                                                <div class="prob-bar-bg">
                                                    <div class="prob-bar-fill" style="width: ${(voter.prob_BJP * 100).toFixed(1)}%; background: ${getPartyColor('BJP')}">
                                                        ${(voter.prob_BJP * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            </div>` : ''}
                                            ${voter.prob_Congress ? `
                                            <div class="prob-item">
                                                <div class="prob-label" style="color: ${getPartyColor('Congress')}">Congress</div>
                                                <div class="prob-bar-bg">
                                                    <div class="prob-bar-fill" style="width: ${(voter.prob_Congress * 100).toFixed(1)}%; background: ${getPartyColor('Congress')}">
                                                        ${(voter.prob_Congress * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            </div>` : ''}
                                            ${voter.prob_AAP ? `
                                            <div class="prob-item">
                                                <div class="prob-label" style="color: ${getPartyColor('AAP')}">AAP</div>
                                                <div class="prob-bar-bg">
                                                    <div class="prob-bar-fill" style="width: ${(voter.prob_AAP * 100).toFixed(1)}%; background: ${getPartyColor('AAP')}">
                                                        ${(voter.prob_AAP * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            </div>` : ''}
                                        </div>
                                    </div>
                                    <div class="detail-box">
                                        <h4>🎯 Prediction Summary</h4>
                                        <div class="value" style="color: ${getPartyColor(voter.predicted_party || 'Others')}">
                                            ${voter.predicted_party || 'Unknown'}
                                        </div>
                                        <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                                            Most likely to vote for <strong>${voter.predicted_party}</strong> with 
                                            <strong>${voter.predicted_party && voter['prob_' + voter.predicted_party] ? (voter['prob_' + voter.predicted_party] * 100).toFixed(1) : 'N/A'}%</strong> confidence
                                        </p>
                                    </div>
                                </div>
                            </td>
                        </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>` : ''}
            
            <!-- Voting Insights -->
            ${props.avg_prob_BJP ? `
            <div class="prediction-summary">
                <h3>🏠 Voting Insights</h3>
                <div style="font-size: 1.1em; line-height: 1.6; color: #333;">
                    <p style="margin-bottom: 15px;">
                        <strong>"People residing in this building tend to favor ${winner}"</strong>
                    </p>
                    <p style="font-style: italic; color: #666; border-left: 4px solid ${getPartyColor(winner)}; padding-left: 15px; margin: 15px 0;">
                        Based on analysis of ${props.voter_count} registered voters in this building, 
                        there's a strong aggregate preference toward <strong style="color: ${getPartyColor(winner)}">${winner}</strong>.
                        ${props.avg_turnout_prob ? `With an expected turnout of ${(props.avg_turnout_prob * 100).toFixed(1)}%, 
                        approximately ${Math.round(props.voter_count * props.avg_turnout_prob)} voters from this building are likely to participate in the election.` : ''}
                    </p>
                </div>
            </div>` : ''}
        </div>
    </div>
    <script>
        function toggleVoterDetails(idx) {
            const detailsRow = document.getElementById('details-' + idx);
            const icon = document.getElementById('icon-' + idx);
            const voterRow = detailsRow.previousElementSibling;
            
            if (detailsRow.classList.contains('show')) {
                detailsRow.classList.remove('show');
                voterRow.classList.remove('expanded');
                icon.textContent = '▶';
            } else {
                detailsRow.classList.add('show');
                voterRow.classList.add('expanded');
                icon.textContent = '▼';
            }
        }
    </script>
</body>
</html>
      `;
      
      // Open in new tab
      const newTab = window.open('', '_blank');
      if (newTab) {
        newTab.document.write(html);
        newTab.document.close();
      } else {
        alert('Please allow popups for this site to view building details');
      }
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
      click: (e) => {
        // Open building details in a new tab
        openBuildingInNewTab(feature, area);
        
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
                console.log(`Rendering building ${index} with style:`, getBuildingStyle(feature, index));
                return getBuildingStyle(feature, index);
              }}
              onEachFeature={onEachBuildingFeature}
              onAdd={() => console.log('Buildings layer added to map')}
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
    </motion.div>
  );
};

export default BoothDetailMap;