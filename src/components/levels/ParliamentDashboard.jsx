import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/index.jsx';
import InteractiveMap from '../InteractiveMap';
import { PartyPerformanceChart, BoothsWonChart, DemographicsChart, ReligionChart } from '../charts';
import { ParliamentStats, QuickStats } from '../stats';
import { MapPin, Users, Vote, Building2 } from 'lucide-react';
import { formatNumber } from '../../shared/utils.js';

const ParliamentDashboard = ({ 
  electoralData, 
  geoJsonData, 
  onAssemblyClick 
}) => {
  if (!electoralData || !electoralData.parliament) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-gray-600">Loading parliament data...</p>
        </div>
      </div>
    );
  }

  const parliamentData = electoralData.parliament;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg">
        <h1 className="text-4xl font-bold mb-2">New Delhi Parliamentary Constituency</h1>
        <p className="text-xl opacity-90">Comprehensive Electoral Analysis Dashboard</p>
        <div className="mt-4 flex justify-center space-x-8">
          <div className="text-center">
            <div className="text-2xl font-bold">{formatNumber(parliamentData.totalVotes)}</div>
            <div className="text-sm opacity-80">Total Votes</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{parliamentData.totalBooths}</div>
            <div className="text-sm opacity-80">Polling Booths</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">10</div>
            <div className="text-sm opacity-80">Assembly Constituencies</div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats data={parliamentData} level="parliament" />

      {/* Main Stats */}
      <ParliamentStats data={parliamentData} />

      {/* Map and Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Interactive Map */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MapPin className="h-5 w-5" />
              <span>Assembly Constituencies Map</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-96">
              <InteractiveMap
                geoJsonData={geoJsonData.assembly}
                electoralData={electoralData}
                level="assembly"
                onFeatureClick={onAssemblyClick}
                center={[28.6139, 77.2090]}
                zoom={11}
              />
            </div>
          </CardContent>
        </Card>

        {/* Party Performance */}
        <div className="space-y-6">
          <PartyPerformanceChart 
            data={parliamentData} 
            title="Vote Share Distribution"
          />
          <BoothsWonChart 
            data={parliamentData} 
            title="Booths Won by Party"
          />
        </div>
      </div>

      {/* Demographics and Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DemographicsChart 
          data={parliamentData} 
          title="Age Distribution"
        />
        <ReligionChart 
          data={parliamentData} 
          title="Religious Composition"
        />
      </div>

      {/* Assembly Constituencies Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Building2 className="h-5 w-5" />
            <span>Assembly Constituencies Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(electoralData.assemblies || {}).map(([name, data]) => (
              <motion.div
                key={name}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all"
                onClick={() => onAssemblyClick && onAssemblyClick(name)}
              >
                <h3 className="font-semibold text-lg mb-2">{name}</h3>
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>AC No:</span>
                    <span className="font-medium">{data.number}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Booths:</span>
                    <span className="font-medium">{data.totalBooths}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Votes:</span>
                    <span className="font-medium">{formatNumber(data.totalVotes)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Leading:</span>
                    <span className="font-medium">
                      {Object.entries(data.boothsWon || {}).reduce((a, b) => 
                        data.boothsWon[a] > data.boothsWon[b[0]] ? a : b[0], 'BJP'
                      )}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Key Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Key Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <Vote className="h-8 w-8 text-blue-600 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Voter Turnout</h3>
              <p className="text-sm text-gray-600">
                High participation across all economic segments
              </p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <Users className="h-8 w-8 text-green-600 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Demographics</h3>
              <p className="text-sm text-gray-600">
                Diverse age and religious composition
              </p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <Building2 className="h-8 w-8 text-purple-600 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Competition</h3>
              <p className="text-sm text-gray-600">
                Multi-party contest across constituencies
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default ParliamentDashboard;