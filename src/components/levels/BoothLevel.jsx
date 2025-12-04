import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  Home, 
  MapPin, 
  Users, 
  Building, 
  Vote, 
  TrendingUp, 
  Clock,
  BarChart3,
  Navigation,
  Info,
  Download,
  Eye,
  UserCheck,
  Search,
  Award
} from 'lucide-react';
import StatCard from '../stats/StatCard.jsx';
import PartyChart from '../charts/PartyChart.jsx';
import DemographicsChart from '../charts/DemographicsChart.jsx';
import AgeGroupChart from '../charts/AgeGroupChart.jsx';
import SelectionDropdown from '../ui/SelectionDropdown.jsx';
import BoothDetailMap from '../BoothDetailMap';
import VoterPredictionCard from '../VoterPredictionCard.jsx';
import { formatNumber, getPartyColor } from '../../shared/utils.js';
import { getBoothCoordinates } from '../../shared/coordinates.js';

const BoothLevel = ({ 
  data, 
  geoData, 
  onNavigateBack, 
  onNavigateHome,
  availableBooths,
  selectedBooth,
  onBoothChange
}) => {
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [buildingAnalytics, setBuildingAnalytics] = useState(null);
  
  // New state for voter predictions
  const [voterPredictions, setVoterPredictions] = useState(null);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [selectedVoterId, setSelectedVoterId] = useState('');
  const [selectedVoterPrediction, setSelectedVoterPrediction] = useState(null);
  const [showIndividualPrediction, setShowIndividualPrediction] = useState(false);
  
  // New state for booth statistics from CSV
  const [boothStatistics, setBoothStatistics] = useState(null);
  const [loadingBoothStats, setLoadingBoothStats] = useState(false);

  // Load building analytics when component mounts
  useEffect(() => {
    const loadBuildingAnalytics = () => {
      // Simulated building analytics data
      const analytics = {
        totalBuildings: 19,
        residentialBuildings: 15,
        commercialBuildings: 2,
        mixedUseBuildings: 2,
        averageBuildingSize: 245,
        estimatedResidents: 890,
        voterDensity: 1.4
      };
      setBuildingAnalytics(analytics);
    };

    loadBuildingAnalytics();
  }, [selectedBooth]);

  const handleBuildingClick = (buildingInfo) => {
    setSelectedBuilding(buildingInfo);
    console.log('Building selected:', buildingInfo);
  };

  // Load booth statistics from Excel (preferred), fallback to CSV predictions
  const loadBoothStatistics = async () => {
    if (!data.AssemblyName || !selectedBooth) return;
    
    try {
      setLoadingBoothStats(true);
      // Prefer Excel stats if available (NewDelhi_Parliamentary_Data.xlsx)
      const excelUrl = `/api/booth-excel-stats/${encodeURIComponent(data.AssemblyName)}/${selectedBooth}`;
      const csvUrl = `/api/booth-statistics/${encodeURIComponent(data.AssemblyName)}/${selectedBooth}`;
      console.log('🔍 Loading booth statistics (Excel preferred):', { assembly: data.AssemblyName, booth: selectedBooth, excelUrl, csvUrl });
      
      let response = await fetch(excelUrl);
      let useCsvFallback = false;
      if (!response.ok) {
        useCsvFallback = true;
        response = await fetch(csvUrl);
      }
      
      if (response.ok) {
        const result = await response.json();
        // Normalize Excel response to existing shape when possible
        const normalized = { ...result };
        // If Excel returned party_probabilities but not predicted_winner/margin, derive minimal fields
        if (!normalized.predicted_winner && normalized.party_probabilities) {
          const entries = Object.entries(normalized.party_probabilities || {});
          const top = entries.sort((a,b)=>b[1]-a[1])[0];
          if (top) normalized.predicted_winner = top[0];
          if (!normalized.expected_votes) normalized.expected_votes = normalized.party_probabilities;
        }
        setBoothStatistics(normalized);
        console.log('✅ Loaded booth statistics:', { source: useCsvFallback ? 'CSV' : 'Excel', result: normalized });
      } else {
        console.error('❌ Failed to load booth statistics:', response.status);
      }
    } catch (error) {
      console.error('❌ Error loading booth statistics:', error);
    } finally {
      setLoadingBoothStats(false);
    }
  };

  // Load voter predictions for the selected booth
  const loadVoterPredictions = async () => {
    if (!data.AssemblyName || !selectedBooth) return;
    
    try {
      setLoadingPredictions(true);
      console.log('🔍 Loading voter predictions for:', {
        assembly: data.AssemblyName,
        booth: selectedBooth,
        url: `/api/voter-predictions/${encodeURIComponent(data.AssemblyName)}/${selectedBooth}`
      });
      
      const response = await fetch(`/api/voter-predictions/${encodeURIComponent(data.AssemblyName)}/${selectedBooth}`);
      
      if (response.ok) {
        const result = await response.json();
        setVoterPredictions(result);
        console.log('✅ Loaded voter predictions:', {
          totalVoters: result.total_voters,
          assembly: result.assembly,
          booth: result.booth,
          sampleVoter: result.voters?.[0]
        });
      } else {
        console.error('❌ Failed to load voter predictions:', response.status);
      }
    } catch (error) {
      console.error('❌ Error loading voter predictions:', error);
    } finally {
      setLoadingPredictions(false);
    }
  };

  // Load individual voter prediction
  const loadIndividualVoterPrediction = async (voterId) => {
    if (!voterId) return;
    
    try {
      console.log('🔍 Loading individual voter prediction for:', voterId);
      const response = await fetch(`/api/voter-prediction/${encodeURIComponent(voterId)}`);
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ Loaded individual voter prediction:', {
          voterId: result.voter?.Voter_ID,
          name: result.voter?.name,
          assembly: result.voter?.assembly_name,
          booth: result.voter?.Booth_ID
        });
        setSelectedVoterPrediction(result.voter);
        setShowIndividualPrediction(true);
      } else {
        const errorData = await response.json();
        console.error('❌ Failed to load individual voter prediction:', response.status, errorData);
        alert(`Failed to load voter prediction: ${errorData.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('❌ Error loading individual voter prediction:', error);
      alert('Error loading voter prediction. Please try again.');
    }
  };

  // Load voter predictions when booth changes
  useEffect(() => {
    if (data?.AssemblyName && selectedBooth) {
      loadVoterPredictions();
      loadBoothStatistics();
    }
  }, [selectedBooth, data?.AssemblyName]);

  // Handle voter ID selection
  const handleVoterIdChange = (voterId) => {
    setSelectedVoterId(voterId);
    if (voterId) {
      loadIndividualVoterPrediction(voterId);
    } else {
      setSelectedVoterPrediction(null);
      setShowIndividualPrediction(false);
    }
  };

  // Early return if no data
  if (!data) return null;

  // Winner (fallback to data.Winner if allocation not available yet)
  let winnerParty = data.Winner || 'Unknown';
  
  // Precise party vote allocation using Largest Remainder so sum == Total_Polled
  const rawRatios = {
    BJP: data.BJP_Ratio || 0,
    AAP: data.AAP_Ratio || 0,
    Congress: data.Congress_Ratio || 0,
    Others: data.Others_Ratio || 0,
    NOTA: data.NOTA_Ratio || 0
  };
  const allocateVotes = (ratios, total) => {
    const parties = Object.keys(ratios);
    const exact = parties.map(p => ({ p, exact: (ratios[p] || 0) * total }));
    const floorVotes = {}; let sumFloors = 0;
    exact.forEach(e => { const f = Math.floor(e.exact); floorVotes[e.p] = f; sumFloors += f; });
    let remaining = total - sumFloors;
    exact.sort((a,b) => (b.exact - Math.floor(b.exact)) - (a.exact - Math.floor(a.exact)));
    let i = 0; while (remaining > 0 && i < exact.length) { floorVotes[exact[i].p] += 1; remaining--; i++; }
    return floorVotes;
  };
  const partyVotes = allocateVotes(rawRatios, data.Total_Polled || 0);
  const totalAllocated = Object.values(partyVotes).reduce((s,v)=>s+v,0);
  if ((data.Total_Polled || 0) !== totalAllocated) {
    console.warn('Vote allocation adjustment', { expected: data.Total_Polled, allocated: totalAllocated });
    const diff = (data.Total_Polled || 0) - totalAllocated;
    if (diff !== 0) {
      const maxParty = Object.entries(partyVotes).sort((a,b)=>b[1]-a[1])[0]?.[0];
      if (maxParty) partyVotes[maxParty] += diff;
    }
  }
  const orderedAllocated = Object.entries(partyVotes).sort((a,b)=>b[1]-a[1]);
  if (orderedAllocated.length) {
    winnerParty = orderedAllocated[0][0];
  }
  const recomputedMargin = orderedAllocated.length >= 2 ? orderedAllocated[0][1] - orderedAllocated[1][1] : (data.Margin || 0);

  // Calculate demographics
  const ageGroups = {
    '18-25': data['Age_18-25_Ratio'] || 0,
    '26-35': data['Age_26-35_Ratio'] || 0,
    '36-45': data['Age_36-45_Ratio'] || 0,
    '46-60': data['Age_46-60_Ratio'] || 0,
    '60+': data['Age_60+_Ratio'] || 0
  };

  const religion = {
    hindu: data.Religion_Hindu_Ratio || 0,
    muslim: data.Religion_Muslim_Ratio || 0,
    sikh: data.Religion_Sikh_Ratio || 0,
    christian: data.Religion_Christian_Ratio || 0,
    buddhist: data.Religion_Buddhist_Ratio || 0,
    jain: data.Religion_Jain_Ratio || 0,
    other: data.Religion_Unknown_Ratio || 0
  };

  const caste = {
    sc: data.Caste_Sc_Ratio || 0,
    obc: data.Caste_Obc_Ratio || 0,
    brahmin: data.Caste_Brahmin_Ratio || 0,
    kshatriya: data.Caste_Kshatriya_Ratio || 0,
    vaishya: data.Caste_Vaishya_Ratio || 0,
    st: data.Caste_St_Ratio || 0
  };

  const genderRatio = {
    male: data.Male_Ratio || 0,
    female: data.Female_Ratio || 0
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className="min-h-screen p-6"
    >
      {/* Header with Navigation */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={onNavigateBack}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <button
              onClick={onNavigateHome}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <Home className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 flex items-center">
                <Building className="w-10 h-10 mr-4 text-orange-600" />
                Booth {data.PartNo}
              </h1>
              <p className="text-gray-600 mt-2">
                {data.Address}
              </p>
              <p className="text-sm text-gray-500">
                Assembly: {data.AssemblyName} | Ward: {data['Ward Name']} | Locality: {data.Locality}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Winner</div>
            <div 
              className="text-2xl font-bold"
              style={{ color: getPartyColor(winnerParty) }}
            >
              {winnerParty}
            </div>
            <div className="text-sm text-gray-500">
              Margin: {Math.round(data.Margin || 0)}
            </div>
          </div>
        </div>

        {/* Booth Selection Dropdown */}
        <div className="mb-6">
          <SelectionDropdown
            options={availableBooths.map(booth => `Booth ${booth.PartNo}`)}
            value={`Booth ${selectedBooth}`}
            onChange={(value) => {
              const boothNo = parseInt(value.split(' ')[1]);
              onBoothChange(boothNo);
            }}
            placeholder="Switch Booth..."
            label="Polling Booth"
            className="max-w-md"
          />
        </div>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Votes"
          value={formatNumber(data.Total_Polled || 0)}
          icon={Vote}
          color="orange"
          subtitle="Votes polled at this booth"
        />
        <StatCard
          title="Total Population"
          value={formatNumber(data.TotalPop || 0)}
          icon={Users}
          color="blue"
          subtitle="Electoral roll size (booth)"
        />
        <StatCard
          title="Winning Party"
          value={winnerParty}
          icon={TrendingUp}
          color="purple"
          subtitle="From electoral data"
          style={{ color: getPartyColor(winnerParty) }}
        />
        <StatCard
          title="Margin"
          value={formatNumber(boothStatistics?.margin ?? recomputedMargin ?? (data.Margin || 0))}
          icon={Clock}
          color="emerald"
          subtitle="Vote difference top two"
        />
      </div>

      {/* Detailed Booth Map */}
      <div className="mb-8">
        {selectedBooth?.toString() === "1" && (data?.AssemblyName === "NEW DELHI" || data?.AssemblyName === "New Delhi") ? (
          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <div className="flex items-center text-blue-800">
              <Building className="w-5 h-5 mr-2" />
              <div>
                <p className="font-semibold">Special Detailed View Available</p>
                <p className="text-sm">This booth has detailed building-level geospatial data</p>
              </div>
            </div>
          </div>
        ) : null}
        
        <BoothDetailMap
          boothNumber={selectedBooth?.toString() || "1"}
          assemblyConstituency={data.AssemblyName || "NEW DELHI"}
          electoralData={{
            Winner: winnerParty,
            TotalVotes: data.Total_Polled || 0,
            Results: partyVotes
          }}
          center={getBoothCoordinates(data.AssemblyName || "NEW DELHI", selectedBooth?.toString() || "1")}
          onBuildingClick={handleBuildingClick}
        />
      </div>

      {/* Selected Building Details */}
      {selectedBuilding && (
        <motion.div
          className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-yellow-500 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <h3 className="text-lg font-bold mb-4 flex items-center text-yellow-700">
            <Eye className="w-5 h-5 mr-2" />
            Selected Building {selectedBuilding.buildingIndex}
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Area:</span>
              <div className="font-bold">{selectedBuilding.area?.toFixed(0)} sq m</div>
            </div>
            <div>
              <span className="text-gray-600">Coordinates:</span>
              <div className="font-bold text-xs">
                {selectedBuilding.coordinates?.[0]?.[0]?.toFixed(6)}, {selectedBuilding.coordinates?.[0]?.[1]?.toFixed(6)}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Est. Floors:</span>
              <div className="font-bold">
                {Math.ceil((selectedBuilding.area || 0) / 100)} floors
              </div>
            </div>
            <div>
              <span className="text-gray-600">Est. Units:</span>
              <div className="font-bold">
                {Math.ceil((selectedBuilding.area || 0) / 50)} units
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Building Analytics */}
      {buildingAnalytics && selectedBooth?.toString() === "1" && (data?.AssemblyName === "NEW DELHI" || data?.AssemblyName === "New Delhi") && (
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <Building className="w-5 h-5 mr-2 text-orange-600" />
            Building Analytics
          </h3>
          <div className="text-sm text-gray-600 mb-4">
            <p>Detailed Building Analysis For Booth 1 </p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded">
              <div className="text-2xl font-bold text-blue-600">
                {buildingAnalytics.residentialBuildings}
              </div>
              <div className="text-gray-600 text-sm">Residential</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded">
              <div className="text-2xl font-bold text-green-600">
                {buildingAnalytics.commercialBuildings}
              </div>
              <div className="text-gray-600 text-sm">Commercial</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded">
              <div className="text-2xl font-bold text-purple-600">
                {buildingAnalytics.mixedUseBuildings}
              </div>
              <div className="text-gray-600 text-sm">Mixed Use</div>
            </div>
            <div className="text-center p-3 bg-orange-50 rounded">
              <div className="text-2xl font-bold text-orange-600">
                {buildingAnalytics.estimatedResidents}
              </div>
              <div className="text-gray-600 text-sm">Est. Residents</div>
            </div>
          </div>
        </div>
      )}

      {/* Swing Factor Analysis Section */}
      {voterPredictions && voterPredictions.voters && voterPredictions.voters.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 p-8 rounded-2xl shadow-2xl mb-8 border-2 border-purple-200"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 flex items-center mb-2">
                <TrendingUp className="w-8 h-8 mr-3 text-purple-600" />
                Voter Alignment Analysis
              </h2>
              <p className="text-sm text-gray-600">Comprehensive swing factor breakdown for Booth {selectedBooth}</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Total Analyzed</div>
              <div className="text-3xl font-bold text-purple-600">{voterPredictions.total_voters}</div>
              <div className="text-xs text-gray-500">voters</div>
            </div>
          </div>

          {(() => {
            // Calculate alignment distribution
            const ALIGN_CORE = 70;
            const ALIGN_LEANING = 40;
            let coreCount = 0, leaningCount = 0, swingCount = 0;
            const partyAlignment = { BJP: { core: 0, leaning: 0, swing: 0 }, Congress: { core: 0, leaning: 0, swing: 0 }, AAP: { core: 0, leaning: 0, swing: 0 }, Others: { core: 0, leaning: 0, swing: 0 } };

            voterPredictions.voters.forEach(v => {
              const probs = [
                { party: 'BJP', prob: v.predictions?.prob_BJP || 0 },
                { party: 'Congress', prob: v.predictions?.prob_Congress || 0 },
                { party: 'AAP', prob: v.predictions?.prob_AAP || 0 },
                { party: 'Others', prob: v.predictions?.prob_Others || 0 }
              ];
              const top = probs.reduce((max, curr) => curr.prob > max.prob ? curr : max);
              let alignment = 'swing';
              if (top.prob >= ALIGN_CORE) { alignment = 'core'; coreCount++; }
              else if (top.prob >= ALIGN_LEANING) { alignment = 'leaning'; leaningCount++; }
              else { swingCount++; }
              
              if (partyAlignment[top.party]) {
                partyAlignment[top.party][alignment]++;
              }
            });

            const total = voterPredictions.total_voters || 1;
            const corePercent = (coreCount / total) * 100;
            const leaningPercent = (leaningCount / total) * 100;
            const swingPercent = (swingCount / total) * 100;

            return (
              <>
                {/* Overall Alignment Breakdown */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  {/* Core Supporters */}
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                    className="bg-white rounded-xl p-6 shadow-lg border-l-4 border-green-500 hover:shadow-xl transition-shadow"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-bold text-green-700">Core Supporters</h3>
                        <p className="text-xs text-gray-500">≥70% confidence</p>
                      </div>
                      <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                        <Award className="w-6 h-6 text-green-600" />
                      </div>
                    </div>
                    <div className="text-4xl font-bold text-green-600 mb-2">{coreCount}</div>
                    <div className="text-sm text-gray-600 mb-4">{corePercent.toFixed(1)}% of total voters</div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div className="bg-green-500 h-3 rounded-full transition-all duration-500" style={{ width: `${corePercent}%` }}></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-3">Strong party allegiance with high prediction confidence</p>
                  </motion.div>

                  {/* Leaning Voters */}
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-white rounded-xl p-6 shadow-lg border-l-4 border-yellow-500 hover:shadow-xl transition-shadow"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-bold text-yellow-700">Leaning Voters</h3>
                        <p className="text-xs text-gray-500">40-69% confidence</p>
                      </div>
                      <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-yellow-600" />
                      </div>
                    </div>
                    <div className="text-4xl font-bold text-yellow-600 mb-2">{leaningCount}</div>
                    <div className="text-sm text-gray-600 mb-4">{leaningPercent.toFixed(1)}% of total voters</div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div className="bg-yellow-500 h-3 rounded-full transition-all duration-500" style={{ width: `${leaningPercent}%` }}></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-3">Moderate preference, persuadable with targeted outreach</p>
                  </motion.div>

                  {/* Swing Voters */}
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="bg-white rounded-xl p-6 shadow-lg border-l-4 border-red-500 hover:shadow-xl transition-shadow"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-bold text-red-700">Swing Voters</h3>
                        <p className="text-xs text-gray-500">&lt;40% confidence</p>
                      </div>
                      <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                        <Users className="w-6 h-6 text-red-600" />
                      </div>
                    </div>
                    <div className="text-4xl font-bold text-red-600 mb-2">{swingCount}</div>
                    <div className="text-sm text-gray-600 mb-4">{swingPercent.toFixed(1)}% of total voters</div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div className="bg-red-500 h-3 rounded-full transition-all duration-500" style={{ width: `${swingPercent}%` }}></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-3">Critical battleground voters with no clear preference</p>
                  </motion.div>
                </div>

                {/* Party-wise Alignment Breakdown */}
                <div className="bg-white rounded-xl p-6 shadow-lg">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-indigo-600" />
                    Party-wise Alignment Distribution
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {Object.entries(partyAlignment).map(([party, counts]) => {
                      const partyTotal = counts.core + counts.leaning + counts.swing;
                      if (partyTotal === 0) return null;
                      return (
                        <div key={party} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <div className="flex items-center justify-between mb-3">
                            <span className="font-bold text-lg" style={{ color: getPartyColor(party) }}>{party}</span>
                            <span className="text-sm font-semibold text-gray-600">{partyTotal} voters</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-green-700">Core</span>
                              <span className="font-bold">{counts.core} ({((counts.core/partyTotal)*100).toFixed(0)}%)</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-green-500 h-2 rounded-full" style={{ width: `${(counts.core/partyTotal)*100}%` }}></div>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-yellow-700">Leaning</span>
                              <span className="font-bold">{counts.leaning} ({((counts.leaning/partyTotal)*100).toFixed(0)}%)</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-yellow-500 h-2 rounded-full" style={{ width: `${(counts.leaning/partyTotal)*100}%` }}></div>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-red-700">Swing</span>
                              <span className="font-bold">{counts.swing} ({((counts.swing/partyTotal)*100).toFixed(0)}%)</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-red-500 h-2 rounded-full" style={{ width: `${(counts.swing/partyTotal)*100}%` }}></div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Strategic Insights */}
                <div className="mt-6 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-200">
                  <h3 className="text-lg font-bold text-indigo-900 mb-4 flex items-center">
                    <Info className="w-5 h-5 mr-2" />
                    Strategic Campaign Insights
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <div className="text-sm font-semibold text-gray-700 mb-2">🎯 Priority Target</div>
                      <div className="text-xs text-gray-600">
                        {swingPercent > 30 ? `High swing voter concentration (${swingPercent.toFixed(0)}%) - intensive ground campaign needed` :
                         swingPercent > 15 ? `Moderate swing voters (${swingPercent.toFixed(0)}%) - strategic outreach recommended` :
                         `Low swing voters (${swingPercent.toFixed(0)}%) - focus on consolidating base`}
                      </div>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <div className="text-sm font-semibold text-gray-700 mb-2">💪 Base Strength</div>
                      <div className="text-xs text-gray-600">
                        {corePercent > 40 ? `Strong core base (${corePercent.toFixed(0)}%) - excellent mobilization potential` :
                         corePercent > 25 ? `Stable core support (${corePercent.toFixed(0)}%) - maintain engagement` :
                         `Weak core base (${corePercent.toFixed(0)}%) - requires base-building efforts`}
                      </div>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <div className="text-sm font-semibold text-gray-700 mb-2">📊 Volatility Index</div>
                      <div className="text-xs text-gray-600">
                        {(leaningPercent + swingPercent) > 60 ? `High volatility (${(leaningPercent + swingPercent).toFixed(0)}%) - outcome unpredictable` :
                         (leaningPercent + swingPercent) > 40 ? `Moderate volatility (${(leaningPercent + swingPercent).toFixed(0)}%) - campaign critical` :
                         `Low volatility (${(leaningPercent + swingPercent).toFixed(0)}%) - stable outcome expected`}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            );
          })()}
        </motion.div>
      )}

      {/* Voter Predictions Section */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
        <h3 className="text-lg font-bold mb-4 flex items-center">
          <UserCheck className="w-5 h-5 mr-2 text-purple-600" />
          Voter Information & AI Predictions - Booth {selectedBooth}
        </h3>
        
        {loadingPredictions && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
              <div className="space-y-2">
                <span className="text-gray-700 font-medium">Loading voter predictions...</span>
                <p className="text-sm text-gray-500">Processing AI predictions for Booth {selectedBooth}</p>
              </div>
            </div>
          </div>
        )}

        {voterPredictions && !loadingPredictions && (
          <div>
            {/* Voter Information & Predictions Table */}
            <div className="overflow-x-auto">
              <h4 className="text-md font-semibold mb-3 flex items-center">
                <BarChart3 className="w-4 h-4 mr-2 text-purple-600" />
                Voter Information Summary
              </h4>
              <table className="min-w-full table-auto">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Voter ID</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Section & Road Name</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Name</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Relation Name</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">House Number</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Age</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Gender</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Relation Type</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Full Address</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Religion</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Caste</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Locality</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {voterPredictions.voters?.slice(0, 20).map((voter, index) => {
                    const voterKey = voter.Voter_ID && voter.Voter_ID.trim() !== '' 
                      ? voter.Voter_ID 
                      : `voter-${index}-${voter.name || 'unknown'}`;
                    return (
                    <tr 
                      key={voterKey} 
                      className={`hover:bg-gray-50 cursor-pointer ${selectedVoterId === voter.Voter_ID ? 'bg-purple-50' : ''}`}
                      onClick={() => handleVoterIdChange(voter.Voter_ID)}
                    >
                      <td className="px-3 py-2 text-xs text-gray-900 border-b font-mono">{voter.Voter_ID}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b max-w-32 truncate" title={voter.section_no_road_name}>{voter.section_no_road_name}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b font-medium">{voter.name}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.relation_name}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.house_number}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.Age}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">
                        <span className={`px-2 py-1 rounded text-xs ${
                          voter.gender === 'Male' ? 'bg-blue-100 text-blue-800' : 
                          voter.gender === 'Female' ? 'bg-pink-100 text-pink-800' : 
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {voter.gender}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.relation_type}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b max-w-40 truncate" title={voter.address}>{voter.address}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.Religion}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b">{voter.Caste}</td>
                      <td className="px-3 py-2 text-xs text-gray-900 border-b max-w-24 truncate" title={voter.Locality}>{voter.Locality}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>

              {voterPredictions.voters && voterPredictions.voters.length > 20 && (
                <div className="text-center py-4 text-sm text-gray-500 bg-gray-50 border-t">
                  <div className="flex items-center justify-center space-x-4">
                    <span>Showing 20 of {voterPredictions.total_voters} voters</span>
                    <span className="text-gray-300">|</span>
                    <span>Click any row for AI predictions</span>
                  </div>
                </div>
              )}
            </div>

            {/* Individual Voter Prediction Selector */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Voter for Detailed Prediction:
              </label>
              <div className="flex space-x-4">
                <select
                  value={selectedVoterId}
                  onChange={(e) => handleVoterIdChange(e.target.value)}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="">Select a Voter ID...</option>
                  {voterPredictions.voters?.map((voter, index) => {
                    const key = voter.Voter_ID && voter.Voter_ID.trim() !== '' ? voter.Voter_ID : `idx-${index}-${voter.name || 'voter'}`;
                    const value = voter.Voter_ID && voter.Voter_ID.trim() !== '' ? voter.Voter_ID : '';
                    return (
                      <option key={key} value={value} disabled={value === ''}>
                        {voter.Voter_ID || '(no id)'} - {voter.name} (Age: {voter.Age}, {voter.gender})
                      </option>
                    );
                  })}
                </select>
                {selectedVoterId && (
                  <button
                    onClick={() => handleVoterIdChange('')}
                    className="px-4 py-2 bg-gray-500 text-white rounded-md text-sm hover:bg-gray-600 transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Individual Voter Prediction Card */}
            {showIndividualPrediction && selectedVoterPrediction && (
              <div className="mb-6">
                <VoterPredictionCard voter={selectedVoterPrediction} />
              </div>
            )}
          </div>
        )}

        {(!voterPredictions || (voterPredictions.voters && voterPredictions.voters.length === 0)) && !loadingPredictions && (
          <div className="text-center py-8 text-gray-500">
            <UserCheck className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>No voter predictions found for Booth {selectedBooth}</p>
            <p className="text-sm">No Predictions available</p>
          </div>
        )}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Party Performance */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Vote className="w-4 h-4 mr-2 text-orange-600" />
            Vote Share
          </h2>
          <PartyChart 
            data={partyVotes} 
            type="pie" 
            showPercentage={true}
            height={180}
          />
          {/* Removed duplicate party list (legend already shown inside PartyChart) */}
        </div>

        {/* Economic Profile (compact) */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <TrendingUp className="w-4 h-4 mr-2 text-orange-600" />
            Economic Profile
          </h2>
          <div className="space-y-3 text-sm">
            <div className="flex items-start justify-between gap-4">
              <span className="text-gray-600 flex-shrink-0">Category</span>
              <span className="font-semibold text-right break-words whitespace-normal">{data.economic_category}</span>
            </div>
            <div className="flex items-start justify-between gap-4">
              <span className="text-gray-600 flex-shrink-0">Land Rate</span>
              <span className="font-semibold text-right">₹{formatNumber(data.land_rate_per_sqm || 0)}</span>
            </div>
            <div className="flex items-start justify-between gap-4">
              <span className="text-gray-600 flex-shrink-0">Construction</span>
              <span className="font-semibold text-right">₹{formatNumber(data.construction_cost_per_sqm || 0)}</span>
            </div>
            <div className="flex items-start justify-between gap-4">
              <span className="text-gray-600 flex-shrink-0">Locality</span>
              <span className="font-semibold text-right break-words whitespace-normal">{data.Locality}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Demographics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-8 mb-8">
        {/* Age Groups */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-orange-600" />
            Age Groups
          </h2>
          <AgeGroupChart 
            data={ageGroups}
            height={180}
          />
        </div>

        {/* Gender Ratio */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-orange-600" />
            Gender Ratio
          </h2>
          <DemographicsChart 
            data={genderRatio}
            type="gender"
          />
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-600">
              Male to Female Ratio: {Math.round((data.MaleToFemaleRatio || 1) * 1000)}:1000
            </p>
          </div>
        </div>

        {/* Religious Composition */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-orange-600" />
            Religion
          </h2>
          <DemographicsChart 
            data={religion}
            type="religion"
          />
        </div>

        {/* Caste Composition */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-orange-600" />
            Caste
          </h2>
          <DemographicsChart 
            data={caste}
            type="caste"
          />
        </div>
      </div>

      {/* Detailed Information */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-6 flex items-center">
          <MapPin className="w-5 h-5 mr-2 text-orange-600" />
          Detailed Information
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Administrative</h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-600">Part No:</span> {data.PartNo}</p>
              <p><span className="text-gray-600">Assembly No:</span> {data.AssemblyNo}</p>
              <p><span className="text-gray-600">Assembly:</span> {data.AssemblyName}</p>
              <p><span className="text-gray-600">Ward No:</span> {data['Ward No.']}</p>
              <p><span className="text-gray-600">Ward:</span> {data['Ward Name']}</p>
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Vote Details</h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-600">Winner:</span> {winnerParty}</p>
              <p><span className="text-gray-600">Margin:</span> {recomputedMargin} votes</p>
              {Math.abs(recomputedMargin - (data.Margin || 0)) > 1 && (
                <p className="text-xs text-yellow-600">(Source file margin: {Math.round(data.Margin || 0)})</p>
              )}
              <p className="mt-1 text-xs text-gray-500">Party Votes: {orderedAllocated.map(([p,v])=>`${p}:${v}`).join(' | ')}</p>
              {(() => {
                const ordered = Object.entries(partyVotes).sort((a,b)=>b[1]-a[1]);
                if (ordered.length >= 2) {
                  const recomputed = ordered[0][1] - ordered[1][1];
                  if (Math.abs(recomputed - (data.Margin || 0)) > 1) {
                    return <p className="text-xs text-yellow-600">Recomputed Margin: {recomputed}</p>;
                  }
                }
                return null;
              })()}
              <p><span className="text-gray-600">NOTA:</span> {Math.round((data.NOTA_Ratio || 0) * (data.Total_Polled || 0))} votes</p>
              <p><span className="text-gray-600">Turnout:</span> {(((data.Total_Polled || 0) / (data.TotalPop || 1)) * 100).toFixed(1)}%</p>
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Location</h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-600">Address:</span></p>
              <p className="text-gray-900">{data.Address}</p>
              <p><span className="text-gray-600">Locality:</span> {data.Locality}</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default BoothLevel;