export const endpoints = {
  health: () => '/api/health',
  modelFeatures: () => '/api/model-features',
  parliamentPreview: (limit = 15) => `/api/parliament-data-preview?limit=${limit}`,
  assemblyPreview: (assembly, limit = 20) =>
    `/api/assembly-data-preview?assembly=${encodeURIComponent(assembly)}&limit=${limit}`,
  boothExcelStats: (assembly, booth) =>
    `/api/booth-excel-stats/${encodeURIComponent(assembly)}/${booth}`,
  boothStatistics: (assembly, booth) =>
    `/api/booth-statistics/${encodeURIComponent(assembly)}/${booth}`,
  voterPredictions: (assembly, booth) =>
    `/api/voter-predictions/${encodeURIComponent(assembly)}/${booth}`,
  voterPrediction: (voterId) => `/api/voter-prediction/${encodeURIComponent(voterId)}`,
  uploadModel: () => '/api/upload-model',
  uploadVoterData: () => '/api/upload-voter-data',
  searchVoter: () => '/api/search-voter',
  predict: () => '/api/predict',
  predictFamily: () => '/api/predict-family'
};
