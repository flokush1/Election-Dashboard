import { useCallback, useEffect, useState } from 'react';
import { apiGet } from '../../../shared/api/client.js';
import { endpoints } from '../../../shared/api/endpoints.js';

export function useBoothStats(assemblyName, boothNumber) {
  const [boothStatistics, setBoothStatistics] = useState(null);
  const [loadingBoothStats, setLoadingBoothStats] = useState(false);

  const loadBoothStatistics = useCallback(async () => {
    if (!assemblyName || !boothNumber) return;
    setLoadingBoothStats(true);
    try {
      try {
        const result = await apiGet(endpoints.boothExcelStats(assemblyName, boothNumber));
        const normalized = { ...result };
        if (!normalized.predicted_winner && normalized.party_probabilities) {
          const top = Object.entries(normalized.party_probabilities).sort((a, b) => b[1] - a[1])[0];
          if (top) normalized.predicted_winner = top[0];
          if (!normalized.expected_votes) normalized.expected_votes = normalized.party_probabilities;
        }
        setBoothStatistics(normalized);
      } catch {
        const result = await apiGet(endpoints.boothStatistics(assemblyName, boothNumber));
        setBoothStatistics(result);
      }
    } catch {
      setBoothStatistics(null);
    } finally {
      setLoadingBoothStats(false);
    }
  }, [assemblyName, boothNumber]);

  useEffect(() => {
    loadBoothStatistics();
  }, [loadBoothStatistics]);

  return { boothStatistics, loadingBoothStats };
}

export function useBoothPredictions(assemblyName, boothNumber) {
  const [voterPredictions, setVoterPredictions] = useState(null);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [selectedVoterId, setSelectedVoterId] = useState('');
  const [selectedVoterPrediction, setSelectedVoterPrediction] = useState(null);
  const [showIndividualPrediction, setShowIndividualPrediction] = useState(false);

  const loadVoterPredictions = useCallback(async () => {
    if (!assemblyName || !boothNumber) return;
    setLoadingPredictions(true);
    try {
      const result = await apiGet(endpoints.voterPredictions(assemblyName, boothNumber));
      setVoterPredictions(result);
    } catch {
      setVoterPredictions(null);
    } finally {
      setLoadingPredictions(false);
    }
  }, [assemblyName, boothNumber]);

  const loadIndividualVoterPrediction = useCallback(async (voterId) => {
    if (!voterId) return;
    try {
      const result = await apiGet(endpoints.voterPrediction(voterId));
      setSelectedVoterPrediction(result.voter);
      setShowIndividualPrediction(true);
    } catch (error) {
      alert(`Failed to load voter prediction: ${error.message || 'Unknown error'}`);
    }
  }, []);

  useEffect(() => {
    loadVoterPredictions();
  }, [loadVoterPredictions]);

  const handleVoterIdChange = (voterId) => {
    setSelectedVoterId(voterId);
    if (voterId) loadIndividualVoterPrediction(voterId);
    else {
      setSelectedVoterPrediction(null);
      setShowIndividualPrediction(false);
    }
  };

  return {
    voterPredictions,
    loadingPredictions,
    selectedVoterId,
    selectedVoterPrediction,
    showIndividualPrediction,
    handleVoterIdChange
  };
}
