import React from 'react';
import StatCard from '../../../components/stats/StatCard.jsx';

const StatsRow = ({ cards = [] }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
    {cards.map((card) => (
      <StatCard key={card.title} {...card} />
    ))}
  </div>
);

export default StatsRow;
