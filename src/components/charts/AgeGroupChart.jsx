import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const AgeGroupChart = ({ data, height = 200 }) => {
  if (!data) return <div className="text-gray-500">No age group data available</div>;

  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  // Convert data to chart format with proper age group labels
  const chartData = Object.entries(data)
    .filter(([key, value]) => value > 0)
    .map(([key, value]) => ({
      ageGroup: key,
      percentage: typeof value === 'number' ? (value < 1 ? value * 100 : value) : 0,
      display: key.includes('-') ? `${key} years` : key === '60+' ? '60+ years' : key
    }))
    .sort((a, b) => {
      // Sort by age order
      const order = ['18-25', '26-35', '36-45', '46-60', '60+'];
      return order.indexOf(a.ageGroup) - order.indexOf(b.ageGroup);
    });

  if (chartData.length === 0) {
    return <div className="text-gray-500">No age group data to display</div>;
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-medium">{data.payload.display}</p>
          <p className="text-sm text-gray-600">
            {data.payload.percentage.toFixed(1)}% of population
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 40, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="display" 
            tick={{ fontSize: 11 }}
            angle={-45}
            textAnchor="end"
            height={60}
            label={{ value: 'Age Groups', position: 'insideBottom', offset: -10 }}
          />
          <YAxis 
            tick={{ fontSize: 11 }}
            label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="percentage" radius={[4, 4, 0, 0]} fill="#3B82F6">
            {chartData.map((entry, index) => (
              <rect key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      {/* Summary Statistics */}
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div className="bg-gray-50 p-2 rounded">
          <span className="text-gray-600">Largest Group:</span>
          <div className="font-medium">
            {chartData.reduce((max, curr) => curr.percentage > max.percentage ? curr : max).display}
          </div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="text-gray-600">Youth (18-35):</span>
          <div className="font-medium">
            {(chartData.filter(d => ['18-25', '26-35'].includes(d.ageGroup))
              .reduce((sum, d) => sum + d.percentage, 0)).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgeGroupChart;