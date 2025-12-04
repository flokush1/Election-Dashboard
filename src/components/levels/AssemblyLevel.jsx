import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Home, MapPin, Users, Building, Vote, TrendingUp } from 'lucide-react';
import InteractiveMap from '../InteractiveMap.jsx';
import StatCard from '../stats/StatCard.jsx';
import PartyChart from '../charts/PartyChart.jsx';
import DemographicsChart from '../charts/DemographicsChart.jsx';
import AgeGroupChart from '../charts/AgeGroupChart.jsx';
import SelectionDropdown from '../ui/SelectionDropdown.jsx';
import { formatNumber, getPartyColor, formatPreviewValue } from '../../shared/utils.js';

// Utility function to filter wards for a specific assembly
const filterWardsForAssembly = (geoData, assemblyName) => {
  if (!geoData || !geoData.features || !assemblyName) {
    return { type: "FeatureCollection", features: [] };
  }
  
  // Normalize assembly name for comparison
  const normalizedAssemblyName = assemblyName.toUpperCase();
  
  const filteredFeatures = geoData.features.filter(feature => {
    const wardAssemblyName = feature.properties?.AC_Name;
    if (!wardAssemblyName) return false;
    
    // Check if ward belongs to this assembly (case-insensitive)
    return wardAssemblyName.toUpperCase() === normalizedAssemblyName;
  });
  
  console.log(`🗺️ Filtering wards for assembly "${assemblyName}":`); 
  console.log(`🗺️ Found ${filteredFeatures.length} wards for this assembly`);
  
  return {
    type: "FeatureCollection",
    features: filteredFeatures
  };
};

const AssemblyLevel = ({ 
  data, 
  wards, 
  geoData, 
  onNavigateToWard, 
  onNavigateBack, 
  onNavigateHome,
  onNavigateToVoterPrediction,
  availableWards,
  selectedAssembly,
  availableAssemblies,
  onAssemblyChange
}) => {
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    if (!selectedAssembly) return;
    const fetchPreview = async () => {
      setPreviewLoading(true);
      setPreviewError(null);
      setPreview(null);
      try {
        const url = `/api/assembly-data-preview?assembly=${encodeURIComponent(selectedAssembly)}&limit=25`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Preview request failed (${res.status})`);
        const json = await res.json();
        if (json.success) {
            setPreview(json);
        } else {
            throw new Error(json.error || 'Unknown error');
        }
      } catch (e) {
        setPreviewError(e.message);
      } finally {
        setPreviewLoading(false);
      }
    };
    fetchPreview();
  }, [selectedAssembly]);

  if (!data) return null;

  const wardsArray = Object.entries(wards || {})
    .filter(([name, ward]) => ward.assembly === selectedAssembly)
    .map(([name, ward]) => ({ name, ...ward }));

  // Determine winning party based on booths won, not total votes
  const winnerParty = data.boothsWon && Object.entries(data.boothsWon)
    .filter(([party]) => party !== 'Tie')
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className="min-h-screen p-6"
    >
      {/* Header with Navigation */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={onNavigateBack}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <button
              onClick={onNavigateHome}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <Home className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 flex items-center">
                <Building className="w-10 h-10 mr-4 text-green-600" />
                {data.name} Assembly
              </h1>
              <p className="text-gray-600 mt-2">
                AC No: {data.number} | {data.totalBooths} booths across {availableWards.length} wards
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Leading Party</div>
            <div 
              className="text-2xl font-bold"
              style={{ color: getPartyColor(winnerParty) }}
            >
              {winnerParty}
            </div>
          </div>
        </div>

        {/* Assembly and Ward Selection Dropdowns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <SelectionDropdown
            options={availableAssemblies}
            value={selectedAssembly}
            onChange={onAssemblyChange}
            placeholder="Switch Assembly..."
            label="Assembly Constituency"
          />
          <SelectionDropdown
            options={availableWards}
            value=""
            onChange={onNavigateToWard}
            placeholder="Select Ward to explore..."
            label="Navigate to Ward"
          />
        </div>

        {/* Voter Prediction Button */}
        {onNavigateToVoterPrediction && (
          <div className="flex justify-center mb-6">
            <button
              onClick={onNavigateToVoterPrediction}
              className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center space-x-2 font-medium"
            >
              <Vote className="w-5 h-5" />
              <span>AI Voter Predictions</span>
            </button>
          </div>
        )}
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Votes"
          value={formatNumber(data.totalVotes)}
          icon={Vote}
          color="blue"
          subtitle="Votes polled"
        />
        <StatCard
          title="Total Population"
          value={formatNumber(data.totalPopulation)}
          icon={Users}
          color="green"
          subtitle="Eligible voters"
        />
        <StatCard
          title="Total Booths"
          value={data.totalBooths}
          icon={MapPin}
          color="purple"
          subtitle="Polling Booths"
        />
        <StatCard
          title="Avg Margin"
          value={Math.round(data.averageMargin || 0)}
          icon={TrendingUp}
          color="orange"
          subtitle="Victory margin"
        />
      </div>

      {/* Ward Performance Statistics */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <MapPin className="w-5 h-5 mr-2 text-green-600" />
          Ward Performance Statistics
        </h2>
        

        {wardsArray && wardsArray.length > 0 ? (
          (() => {
            const wardsWon = {};
            wardsArray.forEach(ward => {
              let winningParty = null;
              if (ward.boothsWon) {
                const boothEntries = Object.entries(ward.boothsWon).filter(([p]) => p !== 'Tie');
                if (boothEntries.length > 0) {
                  winningParty = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
                }
              }
              if (!winningParty && ward.partyVotes) {
                winningParty = Object.entries(ward.partyVotes).reduce((a, b) => a[1] > b[1] ? a : b)[0];
              }
              if (winningParty) {
                wardsWon[winningParty] = (wardsWon[winningParty] || 0) + 1;
              }
            });
            
            const sortedParties = Object.entries(wardsWon)
              .filter(([party, count]) => count > 0)
              .sort(([, a], [, b]) => b - a);
            
            const [leadingParty, leadingCount] = sortedParties[0] || ['None', 0];
            const leadingPercentage = wardsArray.length > 0 ? ((leadingCount / wardsArray.length) * 100).toFixed(1) : '0.0';

            return leadingCount > 0 ? (
              <>
                {/* Leading Party Summary */}
                <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border-l-4 border-green-500">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-gray-600">Leading in ward count: </span>
                      <span 
                        className="font-bold text-xl ml-2"
                        style={{ color: getPartyColor(leadingParty) }}
                      >
                        {leadingParty}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-gray-900">{leadingCount} wards</div>
                      <div className="text-sm text-gray-600">{leadingPercentage}% of total</div>
                    </div>
                  </div>
                </div>

                {/* Party Performance Grid */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {sortedParties.map(([party, count]) => {
                    const percentage = wardsArray.length > 0 ? ((count / wardsArray.length) * 100).toFixed(1) : '0.0';
                    return (
                      <div 
                        key={party}
                        className="text-center p-3 bg-gray-50 rounded-lg border-2 hover:shadow-md transition-shadow"
                        style={{ borderColor: getPartyColor(party) + '40' }}
                      >
                        <div 
                          className="text-2xl font-bold mb-1"
                          style={{ color: getPartyColor(party) }}
                        >
                          {count}
                        </div>
                        <div className="text-xs text-gray-600 mb-1">{percentage}%</div>
                        <div 
                          className="text-sm font-medium px-2 py-1 rounded text-white"
                          style={{ backgroundColor: getPartyColor(party) }}
                        >
                          {party}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <p className="text-gray-500 text-center py-4">No ward performance data available</p>
            );
          })()
        ) : (
          <p className="text-gray-500 text-center py-4">No wards found for this assembly</p>
        )}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Interactive Map */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <MapPin className="w-5 h-5 mr-2 text-green-600" />
            Ward Boundaries Map
          </h2>
          <InteractiveMap
            geoData={filterWardsForAssembly(geoData.ward, selectedAssembly)}
            electoralData={wards}
            level="ward"
            selectedItem={selectedAssembly}
            onItemClick={onNavigateToWard}
          />
        </div>

        {/* Party Performance */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Vote className="w-5 h-5 mr-2 text-green-600" />
            Party Performance
          </h2>
          <PartyChart 
            data={data.partyVotes} 
            type="pie" 
            showPercentage={true}
          />
        </div>
      </div>

      {/* Secondary Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Demographics */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-green-600" />
            Age Groups
          </h2>
          {data.demographics?.ageGroups && (
            <AgeGroupChart 
              data={data.demographics.ageGroups}
              height={180}
            />
          )}
        </div>

        {/* Religious Composition */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-green-600" />
            Religious Composition
          </h2>
          {data.demographics?.religion && (
            <DemographicsChart 
              data={data.demographics.religion}
              type="religion"
              height={180}
            />
          )}
        </div>

        {/* Caste Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-green-600" />
            Caste Distribution
          </h2>
          {data.demographics?.caste && (
            <DemographicsChart 
              data={data.demographics.caste}
              type="caste"
              height={180}
            />
          )}
        </div>

        {/* Economic Categories */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
            Economic Profile
          </h2>
          {data.economics?.categories && (
            <DemographicsChart 
              data={data.economics.categories}
              type="economic"
              height={180}
            />
          )}
        </div>
      </div>

      {/* Wards List */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-6 flex items-center">
          <MapPin className="w-5 h-5 mr-2 text-green-600" />
          Municipal Wards ({availableWards.length})
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {wardsArray.map((ward) => {
            let winningParty = null;
            if (ward.boothsWon) {
              const boothEntries = Object.entries(ward.boothsWon).filter(([p]) => p !== 'Tie');
              if (boothEntries.length > 0) {
                winningParty = boothEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
              }
            }
            if (!winningParty && ward.partyVotes) {
              winningParty = Object.entries(ward.partyVotes).reduce((a, b) => a[1] > b[1] ? a : b)[0];
            }
            
            return (
              <motion.div
                key={ward.name}
                whileHover={{ scale: 1.02 }}
                className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                style={{ borderColor: getPartyColor(winningParty) + '40' }}
                onClick={() => onNavigateToWard(ward.name)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-semibold text-gray-900">{ward.name}</h3>
                  <span 
                    className="text-xs px-2 py-1 rounded-full text-white"
                    style={{ backgroundColor: getPartyColor(winningParty) }}
                  >
                    {winningParty}
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <p>Ward No: {ward.number}</p>
                  <p>Booths: {ward.totalBooths}</p>
                  <p>Votes: {formatNumber(ward.totalVotes)}</p>
                  <p>Margin: {Math.round(ward.averageMargin || 0)}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
        
        {wardsArray.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <MapPin className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No ward data available for {selectedAssembly}</p>
          </div>
        )}
      </div>

        {/* Assembly Excel Data Preview */}
        <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Building className="w-5 h-5 mr-2 text-green-600" />
            {selectedAssembly} Data Preview (Excel rows)
          </h2>
          <p className="text-sm text-gray-500 mb-4">Filtered preview of rows in the parliamentary Excel that match this assembly.</p>
          {previewLoading && <div className="text-gray-600 text-sm">Loading preview...</div>}
          {previewError && <div className="text-red-600 text-sm">Error: {previewError}</div>}
          {preview && (
            <>
              <div className="text-xs text-gray-500 mb-2">Matched rows in file: {preview.matching_rows_found} | Showing first {preview.row_count_preview}</div>
              <div className="overflow-auto border rounded-lg max-h-[480px]">
                <table className="min-w-full text-xs">
                  <thead className="bg-gray-100">
                    <tr>
                      {preview.columns.map(c => (
                        <th key={c.name} className="px-2 py-2 text-left font-semibold border-b whitespace-nowrap">
                          <div>{c.name}</div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        {preview.columns.map(col => (
                          <td key={col.name} className="px-2 py-1 border-b align-top max-w-[200px] truncate" title={String(row[col.name])}>
                            {formatPreviewValue(row[col.name], col.name)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="text-[10px] text-gray-500 p-2">Total columns: {preview.columns.length}</div>
              </div>
              {preview.row_count_preview === 0 && preview.debug && (
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-300 rounded text-xs text-gray-700 space-y-2">
                  <div className="font-semibold text-yellow-800">No rows matched. Debug info:</div>
                  <div><span className="font-medium">Assembly requested:</span> {preview.assembly} (normalized: {preview.assembly_normalized})</div>
                  <div><span className="font-medium">Chosen column:</span> {preview.assembly_column_used} (reason: {preview.assembly_column_reason})</div>
                  <div className="overflow-auto max-h-40">
                    <div className="font-medium mb-1">Candidate scores:</div>
                    <table className="min-w-max border text-[10px]">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-2 py-1 border">Column</th>
                          <th className="px-2 py-1 border">Exact</th>
                          <th className="px-2 py-1 border">Contains</th>
                          <th className="px-2 py-1 border">Text</th>
                        </tr>
                      </thead>
                      <tbody>
                        {preview.debug.candidate_scores.map(cs => (
                          <tr key={cs.column}>
                            <td className="px-2 py-1 border whitespace-nowrap">{cs.column}</td>
                            <td className="px-2 py-1 border text-center">{cs.exact_matches}</td>
                            <td className="px-2 py-1 border text-center">{cs.contains_matches}</td>
                            <td className="px-2 py-1 border text-center">{cs.text_like ? 'Y' : 'N'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <span className="font-medium">Sample values (first 30 unique):</span> {preview.debug.unique_sample_values.join(', ')}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      className="px-3 py-1 bg-blue-600 text-white rounded text-[11px] hover:bg-blue-700"
                      onClick={() => {
                        if (!preview.debug.candidate_scores.length) return;
                        const next = prompt('Manually enter assembly_column to force (see candidate columns above):');
                        if (next) {
                          setPreviewLoading(true);
                          setPreviewError(null);
                          fetch(`/api/assembly-data-preview?assembly=${encodeURIComponent(selectedAssembly)}&limit=25&assembly_column=${encodeURIComponent(next)}`)
                            .then(r => r.json())
                            .then(j => setPreview(j))
                            .catch(e => setPreviewError(e.message))
                            .finally(() => setPreviewLoading(false));
                        }
                      }}
                    >Try Manual Column Override</button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
    </motion.div>
  );
};

export default AssemblyLevel;