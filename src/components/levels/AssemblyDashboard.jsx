import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent, Button } from '../ui/index.jsx';
import InteractiveMap from '../InteractiveMap';
import { PartyPerformanceChart, BoothsWonChart, DemographicsChart, EconomicChart } from '../charts';
import { AssemblyStats, QuickStats } from '../stats';
import { ArrowLeft, MapPin, Users, Building } from 'lucide-react';
import { formatNumber, getPartyColorClass } from '../../shared/utils.js';

const AssemblyDashboard = ({ 
  assemblyName,
  electoralData, 
  geoJsonData, 
  onWardClick,
  onBackClick
}) => {
  if (!assemblyName || !electoralData || !electoralData.assemblies || !electoralData.assemblies[assemblyName]) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-gray-600">Loading assembly data...</p>
        </div>
      </div>
    );
  }

  const assemblyData = electoralData.assemblies[assemblyName];
  const winner = Object.entries(assemblyData.boothsWon || {}).reduce((a, b) => 
    assemblyData.boothsWon[a] > assemblyData.boothsWon[b[0]] ? a : b[0], 'BJP'
  );

  // Get wards in this assembly
  const wardsInAssembly = Object.entries(electoralData.wards || {}).filter(
    ([wardName, wardData]) => wardData.assembly === assemblyName
  );

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
          <span>Back to Parliament</span>
        </Button>
      </div>

      {/* Assembly Header */}
      <div className={`text-center py-8 rounded-lg text-white ${getPartyColorClass(winner)} bg-gradient-to-r`}>
        <h1 className="text-3xl font-bold mb-2">{assemblyName}</h1>
        <p className="text-lg opacity-90">Assembly Constituency No. {assemblyData.number}</p>
        <div className="mt-4 flex justify-center space-x-8">
          <div className="text-center">
            <div className="text-2xl font-bold">{formatNumber(assemblyData.totalVotes)}</div>
            <div className="text-sm opacity-80">Total Votes</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{assemblyData.totalBooths}</div>
            <div className="text-sm opacity-80">Polling Booths</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{winner}</div>
            <div className="text-sm opacity-80">Leading Party</div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats data={assemblyData} level="assembly" />

      {/* Main Stats */}
      <AssemblyStats data={assemblyData} assemblyName={assemblyName} />

      {/* Map and Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Interactive Map - Show wards in this assembly */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MapPin className="h-5 w-5" />
              <span>Ward Boundaries</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-96">
              <InteractiveMap
                geoJsonData={geoJsonData.ward}
                electoralData={electoralData}
                level="ward"
                onFeatureClick={onWardClick}
                center={[28.6139, 77.2090]}
                zoom={12}
              />
            </div>
          </CardContent>
        </Card>

        {/* Party Performance */}
        <div className="space-y-6">
          <PartyPerformanceChart 
            data={assemblyData} 
            title={`Vote Share - ${assemblyName}`}
          />
          <BoothsWonChart 
            data={assemblyData} 
            title="Booths Won by Party"
          />
        </div>
      </div>

      {/* Demographics and Economics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DemographicsChart 
          data={assemblyData} 
          title="Age Distribution"
        />
        <EconomicChart 
          data={assemblyData} 
          title="Economic Categories"
        />
      </div>

      {/* Wards Overview */}
      {wardsInAssembly.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Building className="h-5 w-5" />
              <span>Wards in {assemblyName}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {wardsInAssembly.map(([wardName, wardData]) => (
                <motion.div
                  key={wardName}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                  onClick={() => onWardClick && onWardClick(wardName)}
                >
                  <h3 className="font-semibold text-lg mb-2">{wardName}</h3>
                  <div className="space-y-1 text-sm text-gray-600">
                    <div className="flex justify-between">
                      <span>Ward No:</span>
                      <span className="font-medium">{wardData.number}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Booths:</span>
                      <span className="font-medium">{wardData.totalBooths}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Votes:</span>
                      <span className="font-medium">{formatNumber(wardData.totalVotes)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Leading:</span>
                      <span className="font-medium">
                        {Object.entries(wardData.boothsWon || {}).reduce((a, b) => 
                          wardData.boothsWon[a] > wardData.boothsWon[b[0]] ? a : b[0], 'BJP'
                        )}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Demographics Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Demographic Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <Users className="h-8 w-8 text-blue-600 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Total Population</h3>
              <p className="text-2xl font-bold text-blue-600">
                {formatNumber(assemblyData.demographics?.totalPopulation || 0)}
              </p>
              <p className="text-sm text-gray-600">
                Avg: {assemblyData.demographics?.averagePopulation || 0}/booth
              </p>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-1">
                {((assemblyData.demographics?.genderRatio?.male || 0) * 100).toFixed(1)}%
              </div>
              <h3 className="font-semibold mb-1">Male</h3>
              <p className="text-sm text-gray-600">Gender ratio</p>
            </div>

            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 mb-1">
                {((assemblyData.demographics?.ageGroups?.['26-35'] || 0) * 100).toFixed(1)}%
              </div>
              <h3 className="font-semibold mb-1">Age 26-35</h3>
              <p className="text-sm text-gray-600">Largest group</p>
            </div>

            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600 mb-1">
                ₹{Math.round((assemblyData.economics?.averageLandRate || 0) / 1000)}K
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

export default AssemblyDashboard;