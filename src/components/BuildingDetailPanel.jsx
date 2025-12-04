import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Users, TrendingUp, BarChart3, Home, MapPin, ChevronDown, ChevronUp } from 'lucide-react';

const BuildingDetailPanel = ({ building, onClose }) => {
  const [expandedSections, setExpandedSections] = useState({
    details: true,
    predictions: true,
    voters: true
  });

  if (!building) {
    return (
      <motion.div 
        className="w-96 bg-white shadow-2xl border-l border-gray-200 flex items-center justify-center"
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 100, opacity: 0 }}
      >
        <div className="text-center text-gray-400 p-8">
          <Home className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Click on a building to view details</p>
        </div>
      </motion.div>
    );
  }

  const props = building.properties;
  const hasVoterData = props.voter_count > 0;
  const hasPredictions = props.avg_prob_BJP !== undefined;
  
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

  // Alignment thresholds (mirroring backend / app8 logic)
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

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Calculate area
  const area = props.AREA_SQMTR ? parseFloat(props.AREA_SQMTR).toFixed(0) : 'N/A';

  return (
    <motion.div 
      className="w-96 bg-white shadow-2xl border-l border-gray-200 flex flex-col h-full"
      initial={{ x: 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 100, opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white p-4 flex-shrink-0">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center mb-1">
              <Home className="w-5 h-5 mr-2" />
              <h3 className="text-sm font-semibold text-purple-200">Selected Building</h3>
            </div>
            <h2 className="text-xl font-bold">
              {displayName}
            </h2>
            {!isSpecialBuilding && formatParcelId(props.Parcel_No) !== 'N/A' && (
              <p className="text-xs text-purple-200 mt-1">Parcel: {formatParcelId(props.Parcel_No)}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-white hover:bg-opacity-20 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {hasVoterData && (
          <div className="bg-white bg-opacity-20 rounded-lg p-2 flex items-center justify-between">
            <div className="flex items-center">
              <Users className="w-4 h-4 mr-2" />
              <span className="text-sm font-semibold">{props.voter_count} Voters</span>
            </div>
            {hasPredictions && (
              <div className="text-xs bg-yellow-400 text-purple-900 px-2 py-1 rounded font-bold">
                {winner}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        
        {/* Building Details Section */}
        <div className="border-b border-gray-200">
          <button
            onClick={() => toggleSection('details')}
            className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center">
              <Home className="w-5 h-5 mr-2 text-purple-600" />
              <h3 className="font-semibold text-gray-800">Building Details</h3>
            </div>
            {expandedSections.details ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>
          
          <AnimatePresence>
            {expandedSections.details && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 space-y-2 text-sm">
                  {props.Parcel_No && (
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Parcel ID:</span>
                      <span className="font-semibold text-purple-700">{formatParcelId(props.Parcel_No)}</span>
                    </div>
                  )}
                  {props.NAME && (
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Type:</span>
                      <span className="font-medium text-right">{props.NAME}</span>
                    </div>
                  )}
                  {props.PLOT_NO && (
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Plot No:</span>
                      <span className="font-medium">{props.PLOT_NO}</span>
                    </div>
                  )}
                  {props.Road_No && (
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Road No:</span>
                      <span className="font-medium">{props.Road_No}</span>
                    </div>
                  )}
                  <div className="flex justify-between py-2 border-b border-gray-100">
                    <span className="text-gray-600">Area:</span>
                    <span className="font-medium">{area} sq m</span>
                  </div>
                  {hasVoterData && (
                    <div className="flex justify-between py-2">
                      <span className="text-gray-600">Registered Voters:</span>
                      <span className="font-bold text-blue-600">{props.voter_count}</span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* AI Predictions Section */}
        {hasPredictions && (
          <div className="border-b border-gray-200">
            <button
              onClick={() => toggleSection('predictions')}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-indigo-600" />
                <h3 className="font-semibold text-gray-800">AI Predictions</h3>
              </div>
              {expandedSections.predictions ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            
            <AnimatePresence>
              {expandedSections.predictions && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4">
                    {/* Predicted Winner */}
                    <div className="mb-4 p-3 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border-l-4" style={{ borderColor: getPartyColor(winner) }}>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Predicted Winner:</span>
                        <div className="flex items-center">
                          <TrendingUp className="w-4 h-4 mr-1" style={{ color: getPartyColor(winner) }} />
                          <span className="font-bold text-lg" style={{ color: getPartyColor(winner) }}>
                            {winner}
                          </span>
                        </div>
                      </div>
                      <div className="text-right text-sm text-gray-500 mt-1">
                        {(winnerProb * 100).toFixed(1)}% confidence
                      </div>
                    </div>

                    {/* Alignment Breakdown */}
                    {alignmentSummary && (
                      <div className="mb-4 p-3 bg-purple-50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-600">Alignment Breakdown:</span>
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

                    {/* Party Probabilities */}
                    <div className="space-y-3">
                      {[
                        { name: 'BJP', prob: props.avg_prob_BJP },
                        { name: 'Congress', prob: props.avg_prob_Congress },
                        { name: 'AAP', prob: props.avg_prob_AAP },
                        { name: 'Others', prob: props.avg_prob_Others }
                      ].sort((a, b) => b.prob - a.prob).map((party) => (
                        <div key={party.name}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="font-medium" style={{ color: getPartyColor(party.name) }}>
                              {party.name}
                            </span>
                            <span className="font-bold">{(party.prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="h-2 rounded-full transition-all duration-500"
                              style={{
                                width: `${(party.prob * 100).toFixed(1)}%`,
                                backgroundColor: getPartyColor(party.name)
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Turnout Prediction */}
                    {props.avg_turnout_prob && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Expected Turnout:</span>
                          <span className="font-bold text-blue-600">
                            {(props.avg_turnout_prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Voters List Section */}
        {hasVoterData && props.voters && props.voters.length > 0 && (
          <div className="border-b border-gray-200">
            <button
              onClick={() => toggleSection('voters')}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center">
                <Users className="w-5 h-5 mr-2 text-green-600" />
                <h3 className="font-semibold text-gray-800">
                  Voter Details ({props.voter_count})
                </h3>
              </div>
              {expandedSections.voters ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            
            <AnimatePresence>
              {expandedSections.voters && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 space-y-2">
                    {props.voters.map((voter, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors border border-gray-200">
                        {/* Voter Header */}
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex-1">
                            <div className="font-semibold text-sm text-gray-800">
                              {idx + 1}. {voter.name}
                            </div>
                            <div className="flex gap-3 text-xs text-gray-500 mt-1">
                              <span>Age: {voter.age || 'N/A'}</span>
                              <span>•</span>
                              <span>{voter.gender === 'MALE' ? '♂ Male' : voter.gender === 'FEMALE' ? '♀ Female' : 'N/A'}</span>
                            </div>
                          </div>
                        </div>

                        {/* Voter Predictions */}
                        {voter.predicted_party && (
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs text-gray-600">Likely Vote:</span>
                              <span 
                                className="text-xs font-bold px-2 py-1 rounded"
                                style={{
                                  backgroundColor: `${getPartyColor(voter.predicted_party)}22`,
                                  color: getPartyColor(voter.predicted_party)
                                }}
                              >
                                {voter.predicted_party}
                              </span>
                            </div>

                            {/* Alignment badge */}
                            {(() => {
                              const a = classifyAlignment(voter);
                              const colorMap = { core: '#16a34a', leaning: '#f59e0b', swing: '#dc2626', unknown: '#6b7280' };
                              return (
                                <div className="flex items-center justify-between mb-2 text-xs">
                                  <span className="text-gray-600">Alignment:</span>
                                  <span
                                    className="px-2 py-1 rounded font-semibold"
                                    style={{
                                      backgroundColor: `${colorMap[a.cat]}22`,
                                      color: colorMap[a.cat]
                                    }}
                                  >
                                    {a.cat.charAt(0).toUpperCase() + a.cat.slice(1)} ({(a.prob * 100).toFixed(0)}%)
                                  </span>
                                </div>
                              );
                            })()}
                            
                            {/* Individual probabilities */}
                            <div className="space-y-1">
                              {[
                                { name: 'BJP', prob: voter.prob_BJP },
                                { name: 'Congress', prob: voter.prob_Congress },
                                { name: 'AAP', prob: voter.prob_AAP }
                              ].filter(p => p.prob > 0).sort((a, b) => b.prob - a.prob).map((party) => (
                                <div key={party.name} className="flex items-center justify-between text-xs">
                                  <span style={{ color: getPartyColor(party.name) }}>{party.name}</span>
                                  <div className="flex items-center gap-2 flex-1 ml-2 max-w-[120px]">
                                    <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                                      <div
                                        className="h-1.5 rounded-full"
                                        style={{
                                          width: `${(party.prob * 100).toFixed(0)}%`,
                                          backgroundColor: getPartyColor(party.name)
                                        }}
                                      />
                                    </div>
                                    <span className="text-gray-500 w-10 text-right">
                                      {(party.prob * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>

                            {/* Turnout */}
                            {voter.turnout_prob && (
                              <div className="mt-2 flex items-center justify-between text-xs">
                                <span className="text-gray-600">Turnout Probability:</span>
                                <span className="font-semibold text-blue-600">
                                  {(voter.turnout_prob * 100).toFixed(0)}%
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default BuildingDetailPanel;
