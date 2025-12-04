import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { MapPin, Users, Building, Vote, TrendingUp } from 'lucide-react';
import InteractiveMap from '../InteractiveMap.jsx';
import StatCard from '../stats/StatCard.jsx';
import PartyChart from '../charts/PartyChart.jsx';
import DemographicsChart from '../charts/DemographicsChart.jsx';
import AgeGroupChart from '../charts/AgeGroupChart.jsx';
import SelectionDropdown from '../ui/SelectionDropdown.jsx';
import { formatNumber, getPartyColor, formatPreviewValue } from '../../shared/utils.js';

const ParliamentLevel = ({ 
  data, 
  assemblies, 
  geoData, 
  onNavigateToAssembly,
  onNavigateToVoterPrediction,
  availableAssemblies 
}) => {
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    // Fetch Excel preview once when component mounts
    const fetchPreview = async () => {
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const res = await fetch('/api/parliament-data-preview?limit=15');
        if (!res.ok) {
          throw new Error(`Preview request failed (${res.status})`);
        }
        const json = await res.json();
        if (json.success) {
          setPreview(json);
        } else {
          throw new Error(json.error || 'Unknown error fetching preview');
        }
      } catch (e) {
        setPreviewError(e.message);
      } finally {
        setPreviewLoading(false);
      }
    };
    fetchPreview();
  }, []);

  if (!data) return null;

  const assembliesArray = Object.entries(assemblies || {}).map(([name, data]) => ({
    name,
    ...data
  }));

  const winnerParty = data.partyVotes && Object.entries(data.partyVotes)
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="min-h-screen p-6"
    >
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 flex items-center">
              <Building className="w-10 h-10 mr-4 text-blue-600" />
              {data.name} Parliamentary Constituency
            </h1>
            <p className="text-gray-600 mt-2">
              Comprehensive electoral analysis across assembly constituencies
            </p>
          </div>
          <div className="flex items-center space-x-4">
            {availableAssemblies && availableAssemblies.length > 0 && (
              <SelectionDropdown
                options={availableAssemblies}
                placeholder="Jump to Assembly"
                onChange={onNavigateToAssembly}
              />
            )}
            
            {onNavigateToVoterPrediction && (
              <button
                onClick={onNavigateToVoterPrediction}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center space-x-2 font-medium"
              >
                <Vote className="w-5 h-5" />
                <span>AI Voter Predictions</span>
              </button>
            )}
          </div>
        </div>

        {winnerParty && (
          <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-2">Parliamentary Winner</h2>
                <p className="text-blue-100">
                  Leading party: <span className="font-semibold text-white">{winnerParty}</span>
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold mb-1">
                  {formatNumber(data.partyVotes[winnerParty])}
                </div>
                <div className="text-blue-100">votes</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
        <StatCard
          title="Total Votes"
          value={formatNumber(data.totalVotes)}
          icon={Vote}
          color="blue"
          subtitle="Cast in constituency"
        />
        <StatCard
          title="Total Population"
          value={formatNumber(data.demographics?.totalPopulation || data.totalPopulation)}
          icon={Users}
          color="indigo"
          subtitle="Constituency population"
        />
        <StatCard
          title="Assembly Constituencies"
          value={assembliesArray.length}
          icon={Building}
          color="green"
          subtitle="Administrative divisions"
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

      {/* Assembly Performance Statistics */}
      {assembliesArray && assembliesArray.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Building className="w-5 h-5 mr-2 text-blue-600" />
            Assembly Performance Statistics
          </h2>
          
          {(() => {
            const assembliesWon = {};
            assembliesArray.forEach(assembly => {
              // Determine winning party based on booths won in each assembly
              const winningParty = assembly.boothsWon && Object.entries(assembly.boothsWon)
                .filter(([party]) => party !== 'Tie')
                .reduce((a, b) => a[1] > b[1] ? a : b)[0];
              if (winningParty) {
                assembliesWon[winningParty] = (assembliesWon[winningParty] || 0) + 1;
              }
            });
            
            const sortedParties = Object.entries(assembliesWon)
              .filter(([party, count]) => count > 0)
              .sort(([, a], [, b]) => b - a);
            
            const [leadingParty, leadingCount] = sortedParties[0] || ['None', 0];
            const leadingPercentage = assembliesArray.length > 0 ? ((leadingCount / assembliesArray.length) * 100).toFixed(1) : '0.0';

            return leadingCount > 0 ? (
              <>
                {/* Leading Party Summary */}
                <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border-l-4 border-blue-500">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-gray-600">Leading in assembly count: </span>
                      <span 
                        className="font-bold text-xl ml-2"
                        style={{ color: getPartyColor(leadingParty) }}
                      >
                        {leadingParty}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-gray-900">{leadingCount} assemblies</div>
                      <div className="text-sm text-gray-600">{leadingPercentage}% of total</div>
                    </div>
                  </div>
                </div>

                {/* Party Performance Grid */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {sortedParties.map(([party, count]) => {
                    const percentage = assembliesArray.length > 0 ? ((count / assembliesArray.length) * 100).toFixed(1) : '0.0';
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
              <p className="text-gray-500 text-center py-4">No assembly performance data available</p>
            );
          })()} 
        </div>
      )}

      {/* Booth Performance Statistics */}
      {data.boothsWon && (
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Vote className="w-5 h-5 mr-2 text-green-600" />
            Booth Performance Statistics
          </h2>
          
          {(() => {
            const sortedParties = Object.entries(data.boothsWon)
              .filter(([party, count]) => party !== 'Tie' && count > 0)
              .sort(([, a], [, b]) => b - a);
            
            const [leadingParty, leadingCount] = sortedParties[0] || ['None', 0];
            const leadingPercentage = data.totalBooths > 0 ? ((leadingCount / data.totalBooths) * 100).toFixed(1) : '0.0';

            return leadingCount > 0 ? (
              <>
                {/* Leading Party Summary */}
                <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border-l-4 border-green-500">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-gray-600">Leading in booth count: </span>
                      <span 
                        className="font-bold text-xl ml-2"
                        style={{ color: getPartyColor(leadingParty) }}
                      >
                        {leadingParty}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-gray-900">{leadingCount} booths</div>
                      <div className="text-sm text-gray-600">{leadingPercentage}% of total</div>
                    </div>
                  </div>
                </div>

                {/* Party Performance Grid */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {sortedParties.map(([party, count]) => {
                    const percentage = data.totalBooths > 0 ? ((count / data.totalBooths) * 100).toFixed(1) : '0.0';
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
                  
                  {/* Tie booths if any */}
                  {data.boothsWon.Tie > 0 && (
                    <div className="text-center p-3 bg-gray-50 rounded-lg border-2 border-gray-300">
                      <div className="text-2xl font-bold mb-1 text-gray-600">{data.boothsWon.Tie}</div>
                      <div className="text-xs text-gray-600 mb-1">
                        {((data.boothsWon.Tie / data.totalBooths) * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm font-medium px-2 py-1 rounded bg-gray-400 text-white">
                        Tie
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <p className="text-gray-500 text-center py-4">No booth performance data available</p>
            );
          })()}
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Interactive Map */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <MapPin className="w-5 h-5 mr-2 text-blue-600" />
            Assembly Constituencies Map
          </h2>
          <InteractiveMap
            geoData={geoData.assembly}
            electoralData={assemblies}
            level="assembly"
            selectedItem={null}
            onItemClick={onNavigateToAssembly}
          />
        </div>

        {/* Party Performance */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Vote className="w-5 h-5 mr-2 text-blue-600" />
            Party Performance
          </h2>
          <PartyChart 
            data={data.partyVotes}
            height={220}
          />
        </div>
      </div>

      {/* Demographics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Age Groups */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-purple-600" />
            Age Distribution
          </h2>
          {data.demographics?.ageGroups && (
            <AgeGroupChart 
              data={data.demographics.ageGroups}
              height={220}
            />
          )}
        </div>

        {/* Gender Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-pink-600" />
            Gender Distribution
          </h2>
          {data.demographics?.genderRatio && (
            <PartyChart 
              data={{
                Male: Math.round(data.demographics.genderRatio.male * (data.demographics?.totalPopulation || data.totalPopulation || 0)),
                Female: Math.round(data.demographics.genderRatio.female * (data.demographics?.totalPopulation || data.totalPopulation || 0))
              }}
              height={220}
            />
          )}
        </div>
      </div>

      {/* Assembly Constituencies List */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-6 flex items-center">
          <Building className="w-5 h-5 mr-2 text-blue-600" />
          Assembly Constituencies ({availableAssemblies.length})
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {assembliesArray.map((assembly) => {
            // Determine winning party based on booths won
            const winningParty = assembly.boothsWon && Object.entries(assembly.boothsWon)
              .filter(([party]) => party !== 'Tie')
              .reduce((a, b) => a[1] > b[1] ? a : b)[0];
            
            return (
              <motion.div
                key={assembly.name}
                whileHover={{ scale: 1.02 }}
                className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                style={{ borderColor: getPartyColor(winningParty) + '40' }}
                onClick={() => onNavigateToAssembly(assembly.name)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-semibold text-gray-900">{assembly.name}</h3>
                  <span 
                    className="text-xs px-2 py-1 rounded-full text-white"
                    style={{ backgroundColor: getPartyColor(winningParty) }}
                  >
                    {winningParty}
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <p>AC No: {assembly.number}</p>
                  <p>Booths: {assembly.totalBooths}</p>
                  <p>Votes: {formatNumber(assembly.totalVotes)}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Parliament Excel Data Preview */}
      <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Building className="w-5 h-5 mr-2 text-indigo-600" />
          New Delhi Parliamentary Data Preview
        </h2>
        <p className="text-sm text-gray-500 mb-4">Showing first 15 rows of Excel file for quick inspection.</p>
        {previewLoading && (
          <div className="text-gray-600 text-sm">Loading preview...</div>
        )}
        {previewError && (
          <div className="text-red-600 text-sm">Error: {previewError}</div>
        )}
        {preview && (
          <div className="overflow-auto border rounded-lg">
            <table className="min-w-full text-xs">
              <thead className="bg-gray-100">
                <tr>
                  {preview.columns.map(col => (
                    <th key={col.name} className="px-2 py-2 text-left font-semibold border-b whitespace-nowrap">
                      <div>{col.name}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {preview.columns.map(col => (
                      <td key={col.name} className="px-2 py-1 border-b align-top max-w-[220px] truncate" title={String(row[col.name])}>
                        {formatPreviewValue(row[col.name], col.name)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[10px] text-gray-500 p-2">Total columns: {preview.columns.length}</div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ParliamentLevel;
