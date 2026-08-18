import React from 'react';
import { ArrowLeft, Home } from 'lucide-react';

const LevelHeader = ({
  title,
  subtitle,
  icon: Icon,
  iconClassName = 'text-blue-600',
  onBack,
  onHome,
  actions
}) => (
  <div className="mb-8">
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center space-x-4">
        {onBack && (
          <button onClick={onBack} className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
        )}
        {onHome && (
          <button onClick={onHome} className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow">
            <Home className="w-5 h-5 text-gray-600" />
          </button>
        )}
        <div>
          <h1 className="text-4xl font-bold text-gray-900 flex items-center">
            {Icon && <Icon className={`w-10 h-10 mr-4 ${iconClassName}`} />}
            {title}
          </h1>
          {subtitle && <p className="text-gray-600 mt-2">{subtitle}</p>}
        </div>
      </div>
      {actions}
    </div>
  </div>
);

export default LevelHeader;
