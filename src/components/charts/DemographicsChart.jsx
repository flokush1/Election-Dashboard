import React, { useState } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DemographicsChart = ({ data, type = 'age' }) => {
  if (!data) return <div className="text-gray-500">No data available</div>;

  const getColorScheme = (type) => {
    switch (type) {
      case 'age':
        return ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
      case 'gender':
        return ['#3B82F6', '#EC4899'];
      case 'religion':
        return ['#F97316', '#10B981', '#3B82F6', '#EF4444', '#8B5CF6', '#F59E0B', '#6B7280'];
      case 'caste':
        return ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899'];
      case 'economic':
        return ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'];
      default:
        return ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
    }
  };

  const formatLabel = (key, type) => {
    const labelMaps = {
      age: {
        '18-25': '18-25 years',
        '26-35': '26-35 years',
        '36-45': '36-45 years', 
        '46-60': '46-60 years',
        '60+': '60+ years'
      },
      gender: {
        male: 'Male',
        female: 'Female'
      },
      religion: {
        hindu: 'Hindu',
        muslim: 'Muslim',
        sikh: 'Sikh',
        christian: 'Christian',
        buddhist: 'Buddhist',
        jain: 'Jain',
        other: 'Others'
      },
      caste: {
        sc: 'SC',
        obc: 'OBC',
        brahmin: 'Brahmin',
        kshatriya: 'Kshatriya',
        vaishya: 'Vaishya',
        st: 'ST'
      },
      economic: {
        'LOW INCOME AREAS': 'Low Income',
        'LOWER MIDDLE CLASS': 'Lower Middle',
        'MIDDLE CLASS': 'Middle Class',
        'UPPER MIDDLE CLASS': 'Upper Middle',
        'PREMIUM AREAS': 'Premium'
      }
    };

    return labelMaps[type]?.[key] || key;
  };

  // Convert data to chart format
  const chartData = Object.entries(data)
    .filter(([key, value]) => value > 0)
    .map(([key, value]) => ({
      name: formatLabel(key, type),
      value: typeof value === 'number' ? (value < 1 ? Math.round(value * 100) : Math.round(value)) : 0,
      percentage: typeof value === 'number' ? (value < 1 ? value * 100 : value) : 0
    }))
    .sort((a, b) => b.value - a.value);

  if (chartData.length === 0) {
    return <div className="text-gray-500">No data to display</div>;
  }

  const colors = getColorScheme(type);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-medium text-sm">{data.payload.name}</p>
          <p className="text-xs text-gray-600">
            {type === 'economic' ? `${data.value} booths` : `${data.payload.percentage.toFixed(1)}%`}
          </p>
        </div>
      );
    }
    return null;
  };

  // Use bar chart for economic data (booth counts), pie chart for others (percentages)
  if (type === 'economic') {
    return (
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 40, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 9, fill: '#374151' }}
            axisLine={{ stroke: '#d1d5db' }}
            tickLine={{ stroke: '#d1d5db' }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#374151' }}
            axisLine={{ stroke: '#d1d5db' }}
            tickLine={{ stroke: '#d1d5db' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" fill="#3B82F6" radius={[2, 2, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  const [expanded, setExpanded] = useState(false);

  return (
    <div>
      <ResponsiveContainer width="100%" height={200}>
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
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      
      {/* Legend with expandable list */}
      <div className="mt-2 space-y-1">
        {(expanded ? chartData : chartData.slice(0, 4)).map((entry, index) => (
          <div key={index} className="flex items-center justify-between text-xs">
            <div className="flex items-center">
              <div 
                className="w-2 h-2 rounded-full mr-2 flex-shrink-0"
                style={{ backgroundColor: colors[index % colors.length] }}
              />
              <span className="font-medium truncate">{entry.name}</span>
            </div>
            <span className="font-semibold ml-2 flex-shrink-0">
              {entry.percentage.toFixed(1)}%
            </span>
          </div>
        ))}
        {chartData.length > 4 && (
          <div className="text-xs text-center">
            {!expanded ? (
              <button
                type="button"
                className="text-blue-600 hover:underline"
                onClick={() => setExpanded(true)}
              >
                +{chartData.length - 4} more
              </button>
            ) : (
              <button
                type="button"
                className="text-blue-600 hover:underline"
                onClick={() => setExpanded(false)}
              >
                Show less
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DemographicsChart;