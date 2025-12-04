import React from 'react';
import { motion } from 'framer-motion';
import { 
  User, 
  Home, 
  Calendar, 
  Briefcase, 
  MapPin, 
  Vote, 
  TrendingUp, 
  Award,
  Users,
  Map,
  Building2,
  Heart
} from 'lucide-react';
import { getPartyColor } from '../shared/utils.js';

const VoterPredictionCard = ({ voter }) => {
  if (!voter) return null;

  const predictions = voter.predictions;
  
  // Alignment thresholds (percent scale)
  const ALIGN_CORE = 70;
  const ALIGN_LEANING = 40;
  
  // Find the party with highest probability
  const partyProbs = [
    { party: 'BJP', prob: predictions.prob_BJP },
    { party: 'Congress', prob: predictions.prob_Congress },
    { party: 'AAP', prob: predictions.prob_AAP },
    { party: 'Others', prob: predictions.prob_Others },
    { party: 'NOTA', prob: predictions.prob_NOTA }
  ];
  
  const topParty = partyProbs.reduce((max, current) => 
    current.prob > max.prob ? current : max
  );

  // Calculate confidence level
  const confidence = topParty.prob > 60 ? 'High' : topParty.prob > 40 ? 'Medium' : 'Low';
  const confidenceColor = topParty.prob > 60 ? 'text-green-600' : topParty.prob > 40 ? 'text-yellow-600' : 'text-red-600';

  // Classify alignment (Core / Leaning / Swing)
  let alignmentCat = 'Swing';
  if (topParty.prob >= ALIGN_CORE) alignmentCat = 'Core';
  else if (topParty.prob >= ALIGN_LEANING) alignmentCat = 'Leaning';
  const alignColorMap = { Core: '#16a34a', Leaning: '#f59e0b', Swing: '#dc2626' };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, type: "spring", stiffness: 100 }}
      className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl border border-gray-200 overflow-hidden hover:shadow-2xl transition-all duration-300"
    >
      {/* Header with Gradient */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
              <User className="h-6 w-6" />
            </div>
            <div>
              <h3 className="text-xl font-bold">{voter.name}</h3>
              <p className="text-purple-100 font-mono text-sm">ID: {voter.Voter_ID}</p>
            </div>
          </div>
          
          {/* Turnout Probability Badge */}
          <div className="text-right">
            <div className="text-xs text-purple-100 mb-1">Turnout Probability</div>
            <div className={`px-4 py-2 rounded-full text-lg font-bold bg-white/20 backdrop-blur-sm ${
              predictions.turnout_prob >= 70 ? 'text-green-200' :
              predictions.turnout_prob >= 50 ? 'text-yellow-200' :
              'text-red-200'
            }`}>
              {predictions.turnout_prob.toFixed(1)}%
            </div>
            {/* Alignment Badge */}
            <div className="mt-2">
              <span
                className="px-2 py-1 rounded-full text-xs font-semibold"
                style={{ backgroundColor: `${alignColorMap[alignmentCat]}22`, color: alignColorMap[alignmentCat] }}
                title={`Top party ${topParty.party} at ${topParty.prob.toFixed(1)}%`}
              >
                {alignmentCat} Aligned
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Personal Information Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Basic Info */}
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide flex items-center">
              <Users className="h-4 w-4 mr-2 text-blue-500" />
              Personal Details
            </h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-3 p-2 bg-blue-50 rounded-lg">
                <Calendar className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium">Age: {voter.Age} years</span>
              </div>
              <div className="flex items-center space-x-3 p-2 bg-pink-50 rounded-lg">
                <User className="h-4 w-4 text-pink-600" />
                <span className="text-sm font-medium">{voter.gender}</span>
              </div>
              <div className="flex items-center space-x-3 p-2 bg-green-50 rounded-lg">
                <Heart className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium">{voter.Religion}</span>
              </div>
              <div className="flex items-center space-x-3 p-2 bg-orange-50 rounded-lg">
                <Briefcase className="h-4 w-4 text-orange-600" />
                <span className="text-sm font-medium">{voter.Economic}</span>
              </div>
            </div>
          </div>

          {/* Location Info */}
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide flex items-center">
              <Map className="h-4 w-4 mr-2 text-green-500" />
              Location Details
            </h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-3 p-2 bg-gray-50 rounded-lg">
                <Home className="h-4 w-4 text-gray-600 mt-0.5" />
                <div>
                  <span className="text-sm font-medium block">House: {voter.house_number}</span>
                  <span className="text-xs text-gray-500">{voter.section_no_road_name}</span>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-2 bg-purple-50 rounded-lg">
                <MapPin className="h-4 w-4 text-purple-600" />
                <span className="text-sm font-medium">{voter.Locality}</span>
              </div>
              <div className="flex items-center space-x-3 p-2 bg-indigo-50 rounded-lg">
                <Building2 className="h-4 w-4 text-indigo-600" />
                <span className="text-sm font-medium">Caste: {voter.Caste}</span>
              </div>
              {voter.relation_name && (
                <div className="text-xs text-gray-500 p-2 bg-gray-50 rounded-lg">
                  <strong>Relation:</strong> {voter.relation_name}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Voting Predictions Section */}
        <div className="border-t pt-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-bold text-gray-900 flex items-center">
              <Vote className="h-5 w-5 mr-2 text-purple-600" />
              AI Voting Predictions
            </h4>
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-gray-500" />
              <span className={`text-sm font-semibold ${confidenceColor}`}>
                {confidence} Confidence
              </span>
            </div>
          </div>
          
          <div className="space-y-4">
            {partyProbs.map(({ party, prob }) => (
              <div key={party} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-semibold text-gray-700">{party}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold" style={{ color: getPartyColor(party) }}>
                      {prob.toFixed(1)}%
                    </span>
                    {party === topParty.party && (
                      <Award className="h-4 w-4 text-yellow-500" title="Most Likely Choice" />
                    )}
                  </div>
                </div>
                <div className="relative">
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${prob}%` }}
                      transition={{ duration: 0.8, delay: 0.2 }}
                      className="h-3 rounded-full transition-all duration-300 shadow-sm"
                      style={{
                        backgroundColor: getPartyColor(party),
                        boxShadow: `0 0 10px ${getPartyColor(party)}40`
                      }}
                    />
                  </div>
                  {/* Percentage indicator */}
                  <div 
                    className="absolute top-0 h-3 w-1 bg-white shadow-lg rounded-full transform -translate-x-1/2"
                    style={{ left: `${prob}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Prediction Summary */}
        <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-gray-600">Most Likely Choice:</span>
              <div className="flex items-center space-x-3 mt-1">
                <div
                  className="w-4 h-4 rounded-full shadow-sm"
                  style={{ backgroundColor: getPartyColor(topParty.party) }}
                />
                <span className="text-lg font-bold" style={{ color: getPartyColor(topParty.party) }}>
                  {topParty.party}
                </span>
                <span className="text-sm text-gray-500">
                  ({topParty.prob.toFixed(1)}% probability)
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Prediction Score</div>
              <div className={`text-2xl font-bold ${confidenceColor}`}>
                {((topParty.prob + predictions.turnout_prob) / 2).toFixed(0)}/100
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default VoterPredictionCard;