import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getPartyColor, formatNumber } from '../../shared/utils.js';

const PartyChart = ({ 
  data, 
  type = 'pie', 
  showPercentage = false, 
  height = 250 
}) => {
  if (!data) return <div className="text-gray-500">No data available</div>;

  // Convert data to chart format
  const chartData = Object.entries(data)
    .filter(([party, votes]) => votes > 0)
    .map(([party, votes]) => ({
      name: party,
      value: votes,
      fill: getPartyColor(party)
    }))
    .sort((a, b) => b.value - a.value);

  if (chartData.length === 0) {
    return <div className="text-gray-500">No votes data available</div>;
  }

  const totalVotes = chartData.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      const percentage = ((data.value / totalVotes) * 100).toFixed(1);
      
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-medium text-sm" style={{ color: data.payload.fill }}>
            {data.payload.name}
          </p>
          <p className="text-xs text-gray-600">
            {formatNumber(data.value)} votes ({percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  if (type === 'bar') {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 40, bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 10, fill: '#374151' }}
            axisLine={{ stroke: '#d1d5db' }}
            tickLine={{ stroke: '#d1d5db' }}
            angle={-45}
            textAnchor="end"
            height={50}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#374151' }}
            axisLine={{ stroke: '#d1d5db' }}
            tickLine={{ stroke: '#d1d5db' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" radius={[3, 3, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={false}
            outerRadius={Math.min(height / 3, 70)}
            innerRadius={20}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      
      {/* Compact Legend */}
      <div className="mt-2 space-y-1">
        {chartData.map((entry, index) => {
          const percentage = ((entry.value / totalVotes) * 100).toFixed(1);
          return (
            <div key={index} className="flex items-center justify-between text-xs">
              <div className="flex items-center min-w-0 flex-1">
                <div 
                  className="w-2 h-2 rounded-full mr-2 flex-shrink-0"
                  style={{ backgroundColor: entry.fill }}
                />
                <span className="font-medium truncate">{entry.name}</span>
              </div>
              <div className="text-right ml-2 flex-shrink-0">
                <span className="font-semibold">{formatNumber(entry.value)}</span>
                <span className="text-gray-500 ml-1">({percentage}%)</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PartyChart;