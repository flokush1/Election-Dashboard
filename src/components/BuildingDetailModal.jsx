import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Users, TrendingUp, BarChart3, Home, MapPin } from 'lucide-react';

const BuildingDetailModal = ({ building, onClose }) => {
  if (!building) return null;

  const props = building.properties;
  const hasVoterData = props.voter_count > 0;
  const hasPredictions = props.avg_prob_BJP !== undefined;

  // Alignment thresholds (same as panel / backend logic)
  const ALIGN_CORE = 0.7;
  const ALIGN_LEANING = 0.4;

  // Classify alignment for a voter based on highest party probability
  const classifyAlignment = (voter) => {
    if (!voter) return { cat: 'unknown', prob: 0 };
    const probs = [
      voter.prob_BJP || 0,
      voter.prob_Congress || 0,
      voter.prob_AAP || 0,
      voter.prob_Others || 0
    ];
    const maxProb = Math.max(...probs);
    let cat = 'swing';
    if (maxProb >= ALIGN_CORE) cat = 'core';
    else if (maxProb >= ALIGN_LEANING) cat = 'leaning';
    else cat = 'swing';
    return { cat, prob: maxProb };
  };

  // Aggregate alignment breakdown for building
  const alignmentSummary = (() => {
    if (!props.voters || !props.voters.length) return null;
    let core = 0, leaning = 0, swing = 0;
    props.voters.forEach(v => {
      const a = classifyAlignment(v);
      if (a.cat === 'core') core++;
      else if (a.cat === 'leaning') leaning++;
      else if (a.cat === 'swing') swing++;
    });
    const total = props.voters.length || 1;
    return {
      core,
      leaning,
      swing,
      corePct: (core / total) * 100,
      leaningPct: (leaning / total) * 100,
      swingPct: (swing / total) * 100,
      total
    };
  })();
  
  // Format Parcel ID with / instead of |
  const formatParcelId = (parcelId) => {
    if (!parcelId) return 'N/A';
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
  
  const displayName = getBuildingDisplayName(props);
  const isSpecialBuilding = props.NAME && displayName === props.NAME;
  
  // Get party colors
  const getPartyColor = (party) => {
    const colors = {
      BJP: '#FF9933',
      Congress: '#19AAED',
      AAP: '#0072B0',
      Others: '#808080',
      NOTA: '#000000'
    };
    return colors[party] || '#808080';
  };

  // Find winner
  let winner = 'Unknown';
  let winnerProb = 0;
  if (hasPredictions) {
    const parties = {
      BJP: props.avg_prob_BJP || 0,
      Congress: props.avg_prob_Congress || 0,
      AAP: props.avg_prob_AAP || 0,
      Others: props.avg_prob_Others || 0
    };
    winner = Object.keys(parties).reduce((a, b) => parties[a] > parties[b] ? a : b);
    winnerProb = parties[winner];
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white p-6 relative">
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-2 hover:bg-white hover:bg-opacity-20 rounded-full transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
            
            <div className="flex items-center mb-2">
              <Home className="w-8 h-8 mr-3" />
              <div>
                <h2 className="text-2xl font-bold">
                  {displayName}
                </h2>
                <div className="flex items-center text-purple-100 text-sm mt-1">
                  <MapPin className="w-4 h-4 mr-1" />
                  {isSpecialBuilding && formatParcelId(props.Parcel_No) !== 'N/A' && (
                    <span>Parcel: {formatParcelId(props.Parcel_No)}</span>
                  )}
                  {props.Road_No && <span className={isSpecialBuilding ? "ml-2" : ""}>• Road {props.Road_No}</span>}
                </div>
              </div>
            </div>
            
            {hasVoterData && (
              <div className="flex items-center justify-between mt-4 bg-white bg-opacity-20 rounded-lg p-3">
                <div className="flex items-center">
                  <Users className="w-5 h-5 mr-2" />
                  <span className="text-lg font-semibold">{props.voter_count} Registered Voters</span>
                </div>
                {hasPredictions && (
                  <div className="flex items-center">
                    <TrendingUp className="w-5 h-5 mr-2" />
                    <span className="text-sm">
                      Predicted: <span className="font-bold" style={{ color: '#FFD700' }}>{winner}</span>
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Building Details */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 flex items-center">
                  <Home className="w-5 h-5 mr-2 text-purple-600" />
                  Building Details
                </h3>
                <div className="space-y-2 text-sm">
                  {props.Parcel_No && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Parcel ID:</span>
                      <span className="font-medium text-lg">{formatParcelId(props.Parcel_No)}</span>
                    </div>
                  )}
                  {props.NAME && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Building Type:</span>
                      <span className="font-medium">{props.NAME}</span>
                    </div>
                  )}
                  {props.PLOT_NO && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Plot Number:</span>
                      <span className="font-medium">{props.PLOT_NO}</span>
                    </div>
                  )}
                  {props.Road_No && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Road Number:</span>
                      <span className="font-medium">{props.Road_No}</span>
                    </div>
                  )}
                  {props.AREA_SQMTR && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Area:</span>
                      <span className="font-medium">{parseFloat(props.AREA_SQMTR).toFixed(0)} sq m</span>
                    </div>
                  )}
                </div>
              </div>

              {/* AI Predictions */}
              {hasPredictions && (
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-indigo-600" />
                    AI Vote Predictions
                  </h3>
                  <div className="space-y-3">
                    {[
                      { name: 'BJP', prob: props.avg_prob_BJP },
                      { name: 'Congress', prob: props.avg_prob_Congress },
                      { name: 'AAP', prob: props.avg_prob_AAP },
                      { name: 'Others', prob: props.avg_prob_Others }
                    ].sort((a, b) => b.prob - a.prob).map((party) => (
                      <div key={party.name}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium">{party.name}</span>
                          <span className="font-bold">{(party.prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="h-2 rounded-full transition-all duration-500"
                            style={{
                              width: `${party.prob * 100}%`,
                              backgroundColor: getPartyColor(party.name)
                            }}
                          />
                        </div>
                      </div>
                    ))}
                    {/* Alignment Breakdown */}
                    {alignmentSummary && (
                      <div className="mt-5 p-3 bg-purple-50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-600">Alignment Breakdown</span>
                          <span className="text-xs text-gray-500">{alignmentSummary.total} voters</span>
                        </div>
                        <div className="space-y-2">
                          {[{
                            label: 'Core', value: alignmentSummary.core, pct: alignmentSummary.corePct, color: '#16a34a'
                          }, {
                            label: 'Leaning', value: alignmentSummary.leaning, pct: alignmentSummary.leaningPct, color: '#f59e0b'
                          }, {
                            label: 'Swing', value: alignmentSummary.swing, pct: alignmentSummary.swingPct, color: '#dc2626'
                          }].map(stat => (
                            <div key={stat.label} className="flex items-center text-xs">
                              <div className="w-16 font-medium" style={{ color: stat.color }}>{stat.label}</div>
                              <div className="flex-1 mx-2 bg-gray-200 rounded-full h-2">
                                <div
                                  className="h-2 rounded-full"
                                  style={{ width: `${stat.pct.toFixed(1)}%`, backgroundColor: stat.color }}
                                />
                              </div>
                              <div className="w-20 text-right">
                                {stat.value} ({stat.pct.toFixed(1)}%)
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {props.avg_turnout_prob && (
                      <div className="mt-4 pt-3 border-t border-gray-200">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Avg. Turnout Probability:</span>
                          <span className="font-bold text-green-600">
                            {(props.avg_turnout_prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Voter List */}
            {hasVoterData && props.voters && props.voters.length > 0 && (
              <div className="mt-6 bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 flex items-center">
                  <Users className="w-5 h-5 mr-2 text-purple-600" />
                  Registered Voters ({props.voter_count})
                </h3>
                <div className="max-h-96 overflow-y-auto space-y-3">
                  {props.voters.map((voter, idx) => {
                    // Find voter's predicted party and probabilities
                    let voterParty = 'Unknown';
                    let maxProb = 0;
                    const alignment = classifyAlignment(voter);
                    const alignmentColorMap = { 
                      core: { bg: '#16a34a', text: '#fff', label: 'Core Supporter' },
                      leaning: { bg: '#f59e0b', text: '#fff', label: 'Leaning Voter' },
                      swing: { bg: '#dc2626', text: '#fff', label: 'Swing Voter' },
                      unknown: { bg: '#6b7280', text: '#fff', label: 'Unknown' }
                    };
                    const alignmentStyle = alignmentColorMap[alignment.cat] || alignmentColorMap.unknown;
                    
                    if (voter.prob_BJP !== undefined) {
                      const vParties = {
                        BJP: voter.prob_BJP || 0,
                        Congress: voter.prob_Congress || 0,
                        AAP: voter.prob_AAP || 0,
                        Others: voter.prob_Others || 0
                      };
                      voterParty = Object.keys(vParties).reduce((a, b) => vParties[a] > vParties[b] ? a : b);
                      maxProb = vParties[voterParty];
                    }
                    
                    return (
                      <div 
                        key={idx} 
                        className="bg-gradient-to-r from-gray-50 to-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                      >
                        {/* Header: Name and Alignment Badge */}
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center">
                            <span className="text-gray-500 text-sm mr-3">#{idx + 1}</span>
                            <div>
                              <h4 className="font-semibold text-gray-900">{voter.name}</h4>
                              <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs text-gray-500">
                                  {voter.age ? `${voter.age} yrs` : 'Age N/A'}
                                </span>
                                <span className="text-xs text-gray-400">•</span>
                                <span className="text-xs text-gray-500">
                                  {voter.gender === 'MALE' ? '♂ Male' : voter.gender === 'FEMALE' ? '♀ Female' : 'Gender N/A'}
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          {/* Alignment Badge */}
                          {hasPredictions && (
                            <div
                              className="px-3 py-1.5 rounded-full text-xs font-bold shadow-sm"
                              style={{ 
                                backgroundColor: alignmentStyle.bg,
                                color: alignmentStyle.text
                              }}
                            >
                              {alignmentStyle.label}
                            </div>
                          )}
                        </div>
                        
                        {/* Predictions Section */}
                        {hasPredictions && (
                          <div className="space-y-2">
                            {/* Primary Prediction */}
                            <div className="flex items-center justify-between bg-blue-50 rounded p-2">
                              <span className="text-xs text-gray-600 font-medium">Predicted Vote:</span>
                              <div className="flex items-center gap-2">
                                <span
                                  className="px-3 py-1 rounded text-xs font-bold text-white"
                                  style={{ backgroundColor: getPartyColor(voterParty) }}
                                >
                                  {voterParty}
                                </span>
                                <span className="text-xs text-gray-600">
                                  {(maxProb * 100).toFixed(1)}% confidence
                                </span>
                              </div>
                            </div>
                            
                            {/* Alignment Details */}
                            <div className="flex items-center justify-between bg-purple-50 rounded p-2">
                              <span className="text-xs text-gray-600 font-medium">Voter Alignment:</span>
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold" style={{ color: alignmentStyle.bg }}>
                                  {alignment.cat.toUpperCase()}
                                </span>
                                <span className="text-xs text-gray-600">
                                  {(alignment.prob * 100).toFixed(1)}% commitment
                                </span>
                              </div>
                            </div>
                            
                            {/* Turnout Probability */}
                            {voter.turnout_prob !== undefined && (
                              <div className="flex items-center justify-between bg-green-50 rounded p-2">
                                <span className="text-xs text-gray-600 font-medium">Turnout Probability:</span>
                                <span className="text-xs font-semibold text-green-700">
                                  {(voter.turnout_prob * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default BuildingDetailModal;
