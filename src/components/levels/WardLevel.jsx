import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Home, MapPin, Users, Building, Vote, TrendingUp } from 'lucide-react';
import InteractiveMap from '../InteractiveMap.jsx';
import StatCard from '../stats/StatCard.jsx';
import PartyChart from '../charts/PartyChart.jsx';
import DemographicsChart from '../charts/DemographicsChart.jsx';
import AgeGroupChart from '../charts/AgeGroupChart.jsx';
import SelectionDropdown from '../ui/SelectionDropdown.jsx';
import { formatNumber, getPartyColor } from '../../shared/utils.js';

const WardLevel = ({ 
  data, 
  booths, 
  geoData, 
  onNavigateToBooth, 
  onNavigateBack, 
  onNavigateHome,
  availableBooths,
  selectedWard,
  availableWards,
  onWardChange
}) => {
  if (!data) return null;

  const wardBooths = availableBooths || [];
  
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
                <MapPin className="w-10 h-10 mr-4 text-purple-600" />
                {data.name} Ward
              </h1>
              <p className="text-gray-600 mt-2">
                Ward No: {data.number} | Assembly: {data.assembly} | {data.totalBooths} booths
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

        {/* Ward Selection and Booth Navigation */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <SelectionDropdown
            options={availableWards}
            value={selectedWard}
            onChange={onWardChange}
            placeholder="Switch Ward..."
            label="Municipal Ward"
          />
          <SelectionDropdown
            options={wardBooths.map(booth => `Booth ${booth.PartNo}`)}
            value=""
            onChange={(value) => {
              const boothNo = parseInt(value.split(' ')[1]);
              onNavigateToBooth(boothNo);
            }}
            placeholder="Select Booth to explore..."
            label="Navigate to Booth"
          />
        </div>
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

      {/* Booth Performance Statistics */}
      {data.boothsWon && (
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Vote className="w-5 h-5 mr-2 text-purple-600" />
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
                <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border-l-4 border-purple-500">
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
            <MapPin className="w-5 h-5 mr-2 text-purple-600" />
            Polling Booths Map
          </h2>
          {geoData.booth && geoData.booth.features && geoData.booth.features.length > 0 ? (
            <InteractiveMap
              geoData={geoData.booth}
              electoralData={wardBooths}
              level="booth"
              selectedItem={selectedWard}
              onItemClick={(boothId) => onNavigateToBooth(boothId)}
            />
          ) : (
            <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center">
                <MapPin className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <h3 className="text-lg font-semibold text-gray-600 mb-2">No Booth Map Available</h3>
                <p className="text-gray-500">Booth-level geographic boundaries are not available for this ward.</p>
                <p className="text-sm text-gray-400 mt-2">Use the booth list below to navigate to individual booths.</p>
              </div>
            </div>
          )}
        </div>

        {/* Party Performance */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Vote className="w-5 h-5 mr-2 text-purple-600" />
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
            <Users className="w-5 h-5 mr-2 text-purple-600" />
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
            <Users className="w-5 h-5 mr-2 text-purple-600" />
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
            <Users className="w-5 h-5 mr-2 text-purple-600" />
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
            <TrendingUp className="w-5 h-5 mr-2 text-purple-600" />
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

      {/* Polling Booths List */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-6 flex items-center">
          <Building className="w-5 h-5 mr-2 text-purple-600" />
          Polling Booths ({wardBooths.length})
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {wardBooths.map((booth) => {
            const winningParty = booth.Winner || 'Unknown';
            
            return (
              <motion.div
                key={booth.PartNo}
                whileHover={{ scale: 1.02 }}
                className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                style={{ borderColor: getPartyColor(winningParty) + '40' }}
                onClick={() => onNavigateToBooth(booth.PartNo)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-semibold text-gray-900">Booth {booth.PartNo}</h3>
                  <span 
                    className="text-xs px-2 py-1 rounded-full text-white"
                    style={{ backgroundColor: getPartyColor(winningParty) }}
                  >
                    {winningParty}
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <p className="truncate">{booth.Address}</p>
                  <p>Population: {formatNumber(booth.TotalPop || 0)}</p>
                  <p>Votes: {formatNumber(booth.Total_Polled || 0)}</p>
                  <p>Margin: {Math.round(booth.Margin || 0)}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
        
        {wardBooths.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Building className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No booth data available for {selectedWard}</p>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default WardLevel;