import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent, Badge } from '../ui/index.jsx';
import { getPartyColorClass, formatNumber, formatPercentage, calculateMarginCategory } from '../../shared/utils.js';
import { Users, Vote, Target, TrendingUp, MapPin, Building } from 'lucide-react';

const StatCard = ({ title, value, subtitle, icon: Icon, trend, className = "" }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
  >
    <Card className={`${className} hover:shadow-lg transition-shadow`}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            {subtitle && (
              <p className="text-xs text-gray-500">{subtitle}</p>
            )}
          </div>
          {Icon && (
            <div className="p-3 bg-primary/10 rounded-full">
              <Icon className="h-6 w-6 text-primary" />
            </div>
          )}
        </div>
        {trend && (
          <div className="mt-4 flex items-center space-x-2">
            <TrendingUp className="h-4 w-4 text-green-600" />
            <span className="text-sm text-green-600">{trend}</span>
          </div>
        )}
      </CardContent>
    </Card>
  </motion.div>
);

export const ParliamentStats = ({ data }) => {
  if (!data) return null;

  const winner = Object.entries(data.boothsWon || {}).reduce((a, b) => 
    data.boothsWon[a] > data.boothsWon[b[0]] ? a : b[0], 'BJP'
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <StatCard
        title="Total Votes"
        value={formatNumber(data.totalVotes)}
        subtitle="Across all booths"
        icon={Vote}
      />
      <StatCard
        title="Total Booths"
        value={data.totalBooths}
        subtitle="Polling Booths"
        icon={Building}
      />
      <StatCard
        title="Leading Party"
        value={winner}
        subtitle={`${data.boothsWon[winner]} booths won`}
        className={getPartyColorClass(winner)}
        icon={Target}
      />
      <StatCard
        title="Average Margin"
        value={Math.round(data.averageMargin)}
        subtitle={calculateMarginCategory(data.averageMargin)}
        icon={TrendingUp}
      />
    </div>
  );
};

export const AssemblyStats = ({ data, assemblyName }) => {
  if (!data) return null;

  const winner = Object.entries(data.boothsWon || {}).reduce((a, b) => 
    data.boothsWon[a] > data.boothsWon[b[0]] ? a : b[0], 'BJP'
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <StatCard
        title="Assembly Constituency"
        value={assemblyName}
        subtitle={`AC No. ${data.number}`}
        icon={MapPin}
      />
      <StatCard
        title="Total Votes"
        value={formatNumber(data.totalVotes)}
        subtitle={`${data.totalBooths} booths`}
        icon={Vote}
      />
      <StatCard
        title="Winner"
        value={winner}
        subtitle={`${formatPercentage(data.partyVotes[winner], data.totalVotes)} votes`}
        className={getPartyColorClass(winner)}
        icon={Target}
      />
      <StatCard
        title="Population"
        value={formatNumber(data.demographics?.totalPopulation || 0)}
        subtitle={`Avg: ${data.demographics?.averagePopulation || 0}/booth`}
        icon={Users}
      />
    </div>
  );
};

export const WardStats = ({ data, wardName }) => {
  if (!data) return null;

  const winner = Object.entries(data.boothsWon || {}).reduce((a, b) => 
    data.boothsWon[a] > data.boothsWon[b[0]] ? a : b[0], 'BJP'
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
      <StatCard
        title="Ward"
        value={wardName}
        subtitle={`Ward No. ${data.number} | ${data.assembly}`}
        icon={MapPin}
      />
      <StatCard
        title="Total Votes"
        value={formatNumber(data.totalVotes)}
        subtitle={`${data.totalBooths} booths`}
        icon={Vote}
      />
      <StatCard
        title="Winner"
        value={winner}
        subtitle={`Margin: ${Math.round(data.averageMargin)}`}
        className={getPartyColorClass(winner)}
        icon={Target}
      />
    </div>
  );
};

export const BoothStats = ({ booth }) => {
  if (!booth) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <StatCard
        title="Booth Number"
        value={booth.PartNo}
        subtitle={booth.Address || 'N/A'}
        icon={MapPin}
      />
      <StatCard
        title="Total Polled"
        value={booth.Total_Polled}
        subtitle={`Population: ${booth.TotalPop}`}
        icon={Vote}
      />
      <StatCard
        title="Winner"
        value={booth.Winner}
        subtitle={`Margin: ${booth.Margin}`}
        className={getPartyColorClass(booth.Winner)}
        icon={Target}
      />
      <StatCard
        title="Turnout"
        value={formatPercentage(booth.Total_Polled, booth.TotalPop)}
        subtitle="Voter turnout"
        icon={Users}
      />
    </div>
  );
};

export const QuickStats = ({ data, level }) => {
  if (!data) return null;

  const stats = [];

  if (level === 'parliament') {
    stats.push(
      { label: 'Assemblies', value: '10', color: 'blue' },
      { label: 'Wards', value: '25+', color: 'green' },
      { label: 'Booths', value: data.totalBooths, color: 'purple' }
    );
  } else if (level === 'assembly') {
    stats.push(
      { label: 'Wards', value: 'Multiple', color: 'green' },
      { label: 'Booths', value: data.totalBooths, color: 'purple' },
      { label: 'Avg Margin', value: Math.round(data.averageMargin), color: 'orange' }
    );
  } else if (level === 'ward') {
    stats.push(
      { label: 'Booths', value: data.totalBooths, color: 'purple' },
      { label: 'Population', value: formatNumber(data.demographics?.totalPopulation || 0), color: 'blue' }
    );
  }

  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {stats.map((stat, index) => (
        <Badge key={index} variant="outline" className="px-3 py-1">
          <span className="text-xs text-gray-600">{stat.label}:</span>
          <span className="ml-1 font-semibold">{stat.value}</span>
        </Badge>
      ))}
    </div>
  );
};