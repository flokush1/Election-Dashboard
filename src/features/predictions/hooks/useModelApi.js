import { useEffect, useState } from 'react';
import { apiGet, apiPost, apiRequest } from '../../../shared/api/client.js';
import { endpoints } from '../../../shared/api/endpoints.js';

export function useModelApi() {
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelStatus, setModelStatus] = useState(null);

  const checkApiHealth = async () => {
    try {
      const data = await apiGet(endpoints.health());
      setApiStatus('connected');
      if (data.model_loaded) {
        setModelStatus({
          loaded: true,
          fileName: data.model_file || 'Unknown Model',
          fileSize: 'Already Loaded',
          modelType: 'VoterPredictor',
          features: data.feature_count,
          parties: ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
        });
      }
      return data;
    } catch {
      setApiStatus('error');
      return null;
    }
  };

  useEffect(() => {
    checkApiHealth();
  }, []);

  const uploadModel = (file) => {
    const body = new FormData();
    body.append('model', file);
    return apiRequest(endpoints.uploadModel(), { method: 'POST', body, timeoutMs: 120000 });
  };

  const uploadVoterData = (file) => {
    const body = new FormData();
    body.append('file', file);
    return apiRequest(endpoints.uploadVoterData(), { method: 'POST', body, timeoutMs: 180000 });
  };

  const searchVoter = (voterId) => apiPost(endpoints.searchVoter(), { voter_id: voterId });
  const predictVoter = (voter) => apiPost(endpoints.predict(), voter, { timeoutMs: 60000 });
  const predictFamily = (payload) => apiPost(endpoints.predictFamily(), payload, { timeoutMs: 120000 });

  return {
    apiStatus,
    modelStatus,
    setModelStatus,
    checkApiHealth,
    uploadModel,
    uploadVoterData,
    searchVoter,
    predictVoter,
    predictFamily
  };
}
