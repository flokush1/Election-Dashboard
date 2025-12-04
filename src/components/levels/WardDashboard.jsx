import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent, Button } from '../ui/index.jsx';
import { PartyPerformanceChart, BoothsWonChart, DemographicsChart } from '../charts';
import { WardStats, QuickStats } from '../stats';
import { ArrowLeft, MapPin, Users, Building2 } from 'lucide-react';
import { formatNumber, getPartyColorClass } from '../../shared/utils.js';

const WardDashboard = ({ 
  wardName,
  electoralData, 
  onBoothClick,
  onBackClick
}) => {
  if (!wardName || !electoralData || !electoralData.wards || !electoralData.wards[wardName]) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-gray-600">Loading ward data...</p>
        </div>
      </div>
    );
  }

  const wardData = electoralData.wards[wardName];
  const winner = Object.entries(wardData.boothsWon || {}).reduce((a, b) => 
    wardData.boothsWon[a] > wardData.boothsWon[b[0]] ? a : b[0], 'BJP'
  );

  // Get booths in this ward
  const boothsInWard = electoralData.booths?.filter(
    booth => booth['Ward Name'] === wardName
  ) || [];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Header with Back Button */}
      <div className="flex items-center justify-between">
        <Button 
          variant="outline" 
          onClick={onBackClick}
          className="flex items-center space-x-2"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Assembly</span>
        </Button>
      </div>

      {/* Ward Header */}
      <div className={`text-center py-8 rounded-lg text-white ${getPartyColorClass(winner)} bg-gradient-to-r`}>
        <h1 className="text-3xl font-bold mb-2">{wardName}</h1>
        <p className="text-lg opacity-90">Ward No. {wardData.number} | {wardData.assembly}</p>
        <div className="mt-4 flex justify-center space-x-8">
          <div className="text-center">
            <div className="text-2xl font-bold">{formatNumber(wardData.totalVotes)}</div>
            <div className="text-sm opacity-80">Total Votes</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{wardData.totalBooths}</div>
            <div className="text-sm opacity-80">Polling Booths</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{winner}</div>
            <div className="text-sm opacity-80">Leading Party</div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats data={wardData} level="ward" />

      {/* Main Stats */}
      <WardStats data={wardData} wardName={wardName} />

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PartyPerformanceChart 
          data={wardData} 
          title={`Vote Share - ${wardName}`}
        />
        <BoothsWonChart 
          data={wardData} 
          title="Booths Won by Party"
        />
      </div>

      {/* Demographics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DemographicsChart 
          data={wardData} 
          title="Age Distribution"
        />
        
        <Card>
          <CardHeader>
            <CardTitle>Ward Demographics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded">
                  <div className="text-xl font-bold text-blue-600">
                    {formatNumber(wardData.demographics?.totalPopulation || 0)}
                  </div>
                  <div className="text-sm text-gray-600">Total Population</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded">
                  <div className="text-xl font-bold text-green-600">
                    {wardData.demographics?.averagePopulation || 0}
                  </div>
                  <div className="text-sm text-gray-600">Avg per Booth</div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold">Gender Distribution</h4>
                <div className="flex justify-between text-sm">
                  <span>Male:</span>
                  <span className="font-medium">
                    {((wardData.demographics?.genderRatio?.male || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Female:</span>
                  <span className="font-medium">
                    {((wardData.demographics?.genderRatio?.female || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold">Religious Composition</h4>
                {Object.entries(wardData.demographics?.religion || {}).map(([religion, ratio]) => (
                  <div key={religion} className="flex justify-between text-sm">
                    <span className="capitalize">{religion}:</span>
                    <span className="font-medium">{(ratio * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Polling Booths Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Building2 className="h-5 w-5" />
            <span>Polling Booths in {wardName}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
            {boothsInWard.map((booth) => (
              <motion.div
                key={booth.PartNo}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                onClick={() => onBoothClick && onBoothClick(booth)}
              >
                <h3 className="font-semibold text-lg mb-2">Booth {booth.PartNo}</h3>
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>Address:</span>
                    <span className="font-medium text-xs">{booth.Address?.substring(0, 20)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Polled:</span>
                    <span className="font-medium">{booth.Total_Polled}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Population:</span>
                    <span className="font-medium">{booth.TotalPop}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Winner:</span>
                    <span className={`font-medium ${getPartyColorClass(booth.Winner)}`}>
                      {booth.Winner}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Margin:</span>
                    <span className="font-medium">{booth.Margin}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          
          {boothsInWard.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Building2 className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No booth data available for this ward</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Ward Performance Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 mb-1">
                {Math.round(wardData.averageMargin || 0)}
              </div>
              <h3 className="font-semibold mb-1">Average Margin</h3>
              <p className="text-sm text-gray-600">Victory margin</p>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-1">
                {((wardData.totalVotes / (wardData.demographics?.totalPopulation || 1)) * 100).toFixed(1)}%
              </div>
              <h3 className="font-semibold mb-1">Turnout Rate</h3>
              <p className="text-sm text-gray-600">Voter participation</p>
            </div>

            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 mb-1">
                {wardData.economics?.averageLandRate ? 
                  `₹${Math.round(wardData.economics.averageLandRate / 1000)}K` : 'N/A'
                }
              </div>
              <h3 className="font-semibold mb-1">Avg Land Rate</h3>
              <p className="text-sm text-gray-600">Per sq meter</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default WardDashboard;