import React from 'react';
import { motion } from 'framer-motion';

const StatCard = ({ 
  title, 
  value, 
  icon: Icon, 
  color = 'blue', 
  subtitle = '', 
  trend = null,
  className = '' 
}) => {
  const colorMap = {
    blue: 'bg-blue-500 text-blue-600 bg-blue-50',
    green: 'bg-green-500 text-green-600 bg-green-50',
    purple: 'bg-purple-500 text-purple-600 bg-purple-50',
    orange: 'bg-orange-500 text-orange-600 bg-orange-50',
    red: 'bg-red-500 text-red-600 bg-red-50',
    yellow: 'bg-yellow-500 text-yellow-600 bg-yellow-50'
  };

  const [bgColor, textColor, cardBg] = colorMap[color]?.split(' ') || colorMap.blue.split(' ');

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={`${cardBg} rounded-xl p-6 shadow-sm border border-gray-200 ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <div className={`flex items-center mt-2 text-sm ${trend.positive ? 'text-green-600' : 'text-red-600'}`}>
              <span>{trend.value}</span>
              <span className="ml-1">{trend.positive ? '↗' : '↘'}</span>
            </div>
          )}
        </div>
        <div className={`${bgColor} p-3 rounded-lg`}>
          <Icon className={`w-6 h-6 text-white`} />
        </div>
      </div>
    </motion.div>
  );
};

export default StatCard;