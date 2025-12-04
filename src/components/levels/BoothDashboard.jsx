import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '../ui/index.jsx';
import { BoothStats } from '../stats';
import { ArrowLeft, MapPin, Users, Vote, Building, DollarSign } from 'lucide-react';
import { formatNumber, formatPercentage, getPartyColorClass, getPartyColor } from '../../shared/utils.js';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const BoothDashboard = ({ 
  booth,
  onBackClick
}) => {
  if (!booth) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-gray-600">Loading booth data...</p>
        </div>
      </div>
    );
  }

  // Prepare party vote data
  const partyVotes = [
    { party: 'BJP', votes: Math.round(booth.BJP_Ratio * booth.Total_Polled), ratio: booth.BJP_Ratio },
    { party: 'AAP', votes: Math.round(booth.AAP_Ratio * booth.Total_Polled), ratio: booth.AAP_Ratio },
    { party: 'Congress', votes: Math.round(booth.Congress_Ratio * booth.Total_Polled), ratio: booth.Congress_Ratio },
    { party: 'Others', votes: Math.round(booth.Others_Ratio * booth.Total_Polled), ratio: booth.Others_Ratio },
    { party: 'NOTA', votes: Math.round(booth.NOTA_Ratio * booth.Total_Polled), ratio: booth.NOTA_Ratio }
  ].filter(item => item.votes > 0);

  // Age distribution data
  const ageData = [
    { ageGroup: '18-25', percentage: (booth['Age_18-25_Ratio'] * 100).toFixed(1) },
    { ageGroup: '26-35', percentage: (booth['Age_26-35_Ratio'] * 100).toFixed(1) },
    { ageGroup: '36-45', percentage: (booth['Age_36-45_Ratio'] * 100).toFixed(1) },
    { ageGroup: '46-60', percentage: (booth['Age_46-60_Ratio'] * 100).toFixed(1) },
    { ageGroup: '60+', percentage: (booth['Age_60+_Ratio'] * 100).toFixed(1) }
  ];

  // Religion data
  const religionData = [
    { religion: 'Hindu', percentage: (booth.Religion_Hindu_Ratio * 100).toFixed(1) },
    { religion: 'Muslim', percentage: (booth.Religion_Muslim_Ratio * 100).toFixed(1) },
    { religion: 'Sikh', percentage: (booth.Religion_Sikh_Ratio * 100).toFixed(1) },
    { religion: 'Christian', percentage: (booth.Religion_Christian_Ratio * 100).toFixed(1) },
    { religion: 'Others', percentage: ((booth.Religion_Buddhist_Ratio + booth.Religion_Jain_Ratio + booth.Religion_Unknown_Ratio) * 100).toFixed(1) }
  ].filter(item => parseFloat(item.percentage) > 0);

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
          <span>Back to Ward</span>
        </Button>
      </div>

      {/* Booth Header */}
      <div className={`text-center py-8 rounded-lg text-white ${getPartyColorClass(booth.Winner)} bg-gradient-to-r`}>
        <h1 className="text-3xl font-bold mb-2">Polling Booth {booth.PartNo}</h1>
        <p className="text-lg opacity-90">{booth.AssemblyName} | {booth['Ward Name'] || 'N/A'}</p>
        <div className="mt-4 flex justify-center space-x-8">
          <div className="text-center">
            <div className="text-2xl font-bold">{booth.Total_Polled}</div>
            <div className="text-sm opacity-80">Votes Polled</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{booth.TotalPop}</div>
            <div className="text-sm opacity-80">Population</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{booth.Winner}</div>
            <div className="text-sm opacity-80">Winner</div>
          </div>
        </div>
      </div>

      {/* Booth Stats */}
      <BoothStats booth={booth} />

      {/* Key Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MapPin className="h-5 w-5" />
            <span>Booth Information</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Address:</span>
                <span className="font-medium text-right">{booth.Address}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Assembly No:</span>
                <span className="font-medium">{booth.AssemblyNo}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Ward No:</span>
                <span className="font-medium">{booth['Ward No.'] || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Economic Category:</span>
                <Badge variant="outline">{booth.economic_category}</Badge>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Victory Margin:</span>
                <span className="font-medium">{booth.Margin} votes</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Turnout:</span>
                <span className="font-medium">{formatPercentage(booth.Total_Polled, booth.TotalPop)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Land Rate:</span>
                <span className="font-medium">₹{formatNumber(booth.land_rate_per_sqm)}/sq.m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Construction Cost:</span>
                <span className="font-medium">₹{formatNumber(booth.construction_cost_per_sqm)}/sq.m</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Election Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Vote Share Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Vote Share Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={partyVotes}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ party, ratio }) => `${party} ${(ratio * 100).toFixed(1)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="votes"
                  >
                    {partyVotes.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getPartyColor(entry.party)} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value, name) => [value, 'Votes']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Party Performance Table */}
        <Card>
          <CardHeader>
            <CardTitle>Party Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {partyVotes.map((party) => (
                <div key={party.party} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: getPartyColor(party.party) }}
                    ></div>
                    <span className="font-medium">{party.party}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{party.votes}</div>
                    <div className="text-sm text-gray-600">{(party.ratio * 100).toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Demographics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Age Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Age Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ageData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="ageGroup" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, 'Population']} />
                  <Bar dataKey="percentage" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Religious Composition */}
        <Card>
          <CardHeader>
            <CardTitle>Religious Composition</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {religionData.map((religion, index) => (
                <div key={religion.religion} className="flex items-center justify-between">
                  <span className="font-medium">{religion.religion}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${religion.percentage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium w-12 text-right">{religion.percentage}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Demographics */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Demographics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Gender */}
            <div className="space-y-2">
              <h4 className="font-semibold flex items-center space-x-2">
                <Users className="h-4 w-4" />
                <span>Gender Distribution</span>
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Male:</span>
                  <span className="font-medium">{(booth.Male_Ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Female:</span>
                  <span className="font-medium">{(booth.Female_Ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Male-Female Ratio:</span>
                  <span className="font-medium">{booth.MaleToFemaleRatio?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>

            {/* Caste */}
            <div className="space-y-2">
              <h4 className="font-semibold">Caste Distribution</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>SC:</span>
                  <span className="font-medium">{(booth.Caste_Sc_Ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>OBC:</span>
                  <span className="font-medium">{(booth.Caste_Obc_Ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>General:</span>
                  <span className="font-medium">{((booth.Caste_Brahmin_Ratio + booth.Caste_Kshatriya_Ratio + booth.Caste_Vaishya_Ratio) * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Economic */}
            <div className="space-y-2">
              <h4 className="font-semibold flex items-center space-x-2">
                <DollarSign className="h-4 w-4" />
                <span>Economic Profile</span>
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Category:</span>
                  <span className="font-medium text-xs">{booth.economic_category}</span>
                </div>
                <div className="flex justify-between">
                  <span>Land Rate:</span>
                  <span className="font-medium">₹{Math.round(booth.land_rate_per_sqm/1000)}K</span>
                </div>
                <div className="flex justify-between">
                  <span>Construction:</span>
                  <span className="font-medium">₹{Math.round(booth.construction_cost_per_sqm/1000)}K</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default BoothDashboard;