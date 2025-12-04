import React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/index.jsx';
import { getPartyColor, formatNumber, formatPercentage } from '../../shared/utils.js';

const ChartContainer = ({ title, children, className = "", height = "h-64" }) => (
  <Card className={className}>
    <CardHeader>
      <CardTitle className="text-lg font-semibold">{title}</CardTitle>
    </CardHeader>
    <CardContent>
      <div className={height}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </CardContent>
  </Card>
);

export const PartyPerformanceChart = ({ data, title = "Party Performance" }) => {
  if (!data || !data.partyVotes) return null;

  const chartData = Object.entries(data.partyVotes)
    .map(([party, votes]) => ({
      party,
      votes,
      percentage: (votes / data.totalVotes) * 100
    }))
    .filter(item => item.votes > 0)
    .sort((a, b) => b.votes - a.votes);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="60%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={false}
                outerRadius={60}
                innerRadius={20}
                fill="#8884d8"
                dataKey="votes"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getPartyColor(entry.party)} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value, name) => [formatNumber(value), 'Votes']}
                labelFormatter={() => ''}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-2 space-y-1">
            {chartData.map((entry, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getPartyColor(entry.party) }}
                  />
                  <span className="font-medium">{entry.party}</span>
                </div>
                <div className="text-right">
                  <span className="font-semibold">{formatNumber(entry.votes)}</span>
                  <span className="text-gray-500 ml-1">({entry.percentage.toFixed(1)}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const BoothsWonChart = ({ data, title = "Booths Won by Party" }) => {
  if (!data || !data.boothsWon) return null;

  const chartData = Object.entries(data.boothsWon)
    .map(([party, booths]) => ({
      party,
      booths,
      percentage: (booths / data.totalBooths) * 100
    }))
    .filter(item => item.booths > 0)
    .sort((a, b) => b.booths - a.booths);

  return (
    <ChartContainer title={title} height="h-64">
      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="party" 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
          angle={-45}
          textAnchor="end"
          height={40}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <Tooltip 
          formatter={(value, name) => [value, 'Booths']}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar dataKey="booths" radius={[2, 2, 0, 0]}>
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getPartyColor(entry.party)} />
          ))}
        </Bar>
      </BarChart>
    </ChartContainer>
  );
};

export const DemographicsChart = ({ data, title = "Age Distribution" }) => {
  if (!data || !data.demographics || !data.demographics.ageGroups) return null;

  const chartData = Object.entries(data.demographics.ageGroups)
    .map(([ageGroup, ratio]) => ({
      ageGroup: ageGroup.replace(/([0-9]+)/g, '$1 '),
      percentage: parseFloat((ratio * 100).toFixed(1))
    }))
    .sort((a, b) => {
      const getAge = (group) => parseInt(group.match(/\d+/)?.[0] || '0');
      return getAge(a.ageGroup) - getAge(b.ageGroup);
    });

  return (
    <ChartContainer title={title} height="h-64">
      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="ageGroup" 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
          angle={-45}
          textAnchor="end"
          height={40}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <Tooltip 
          formatter={(value) => [`${value}%`, 'Population']}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar dataKey="percentage" fill="#3B82F6" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ChartContainer>
  );
};

export const ReligionChart = ({ data, title = "Religious Composition" }) => {
  if (!data || !data.demographics || !data.demographics.religion) return null;

  const colors = {
    hindu: '#FF6B35',
    muslim: '#4ECDC4', 
    sikh: '#45B7D1',
    christian: '#96CEB4',
    jain: '#E74C3C',
    buddhist: '#9B59B6',
    other: '#FECA57'
  };

  const chartData = Object.entries(data.demographics.religion)
    .map(([religion, ratio]) => ({
      religion: religion.charAt(0).toUpperCase() + religion.slice(1),
      percentage: parseFloat((ratio * 100).toFixed(1))
    }))
    .filter(item => item.percentage > 0)
    .sort((a, b) => b.percentage - a.percentage);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="60%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={false}
                outerRadius={60}
                innerRadius={20}
                fill="#8884d8"
                dataKey="percentage"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[entry.religion.toLowerCase()] || '#8884d8'} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value) => [`${value}%`, 'Population']}
                labelFormatter={() => ''}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-2 space-y-1">
            {chartData.slice(0, 4).map((entry, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: colors[entry.religion.toLowerCase()] || '#8884d8' }}
                  />
                  <span className="font-medium">{entry.religion}</span>
                </div>
                <span className="font-semibold">{entry.percentage}%</span>
              </div>
            ))}
            {chartData.length > 4 && (
              <div className="text-xs text-gray-500 text-center">
                +{chartData.length - 4} more
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const EconomicChart = ({ data, title = "Economic Categories" }) => {
  if (!data || !data.economics || !data.economics.categories) return null;

  const chartData = Object.entries(data.economics.categories)
    .map(([category, count]) => ({
      category: category.replace(/\s+/g, ' ').replace(/([A-Z])/g, ' $1').trim(),
      count
    }))
    .filter(item => item.count > 0)
    .sort((a, b) => b.count - a.count);

  return (
    <ChartContainer title={title} height="h-64">
      <BarChart data={chartData} layout="horizontal" margin={{ top: 10, right: 20, left: 80, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          type="number" 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <YAxis 
          dataKey="category" 
          type="category" 
          width={80}
          tick={{ fontSize: 9, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <Tooltip 
          formatter={(value) => [value, 'Booths']}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar dataKey="count" fill="#10B981" radius={[0, 2, 2, 0]} />
      </BarChart>
    </ChartContainer>
  );
};

export const TrendChart = ({ data, title = "Victory Margin Distribution" }) => {
  if (!data || !data.booths) return null;

  // Sample data for margin distribution - this would be calculated from actual booth data
  const marginRanges = [
    { range: '0-50', count: Math.floor(Math.random() * 20) + 5, label: 'Very Close (0-50)' },
    { range: '51-100', count: Math.floor(Math.random() * 30) + 10, label: 'Close (51-100)' },
    { range: '101-200', count: Math.floor(Math.random() * 40) + 15, label: 'Comfortable (101-200)' },
    { range: '201+', count: Math.floor(Math.random() * 50) + 20, label: 'Safe (201+)' }
  ];

  return (
    <ChartContainer title={title} height="h-64">
      <BarChart data={marginRanges} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="range" 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: '#374151' }}
          axisLine={{ stroke: '#d1d5db' }}
          tickLine={{ stroke: '#d1d5db' }}
        />
        <Tooltip 
          formatter={(value) => [value, 'Booths']}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar dataKey="count" fill="#8B5CF6" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ChartContainer>
  );
};