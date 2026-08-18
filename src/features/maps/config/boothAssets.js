export const BOOTH_ASSETS = {
  'NEW DELHI|1': {
    boundaryUrl: '/data/geospatial/New_Delhi_Booth_Data.geojson',
    buildingsUrl: '/data/geospatial/New_Delhi_Booth_Buildings.geojson'
  },
  'NEW DELHI|103': {
    boundaryUrl: '/data/geospatial/New_Delhi_Booth_Data.geojson',
    buildingsUrl: '/data/geospatial/NewDelhi_BlockE_Booth_103_Plots_With_Predictions.geojson',
    dedicated: true
  },
  'R K PURAM|17': {
    boundaryUrl: '/data/geospatial/RKPuram_Booth_17_Boundary.geojson',
    buildingsUrl: '/data/geospatial/RKPuram_Booth_17_Plots_With_Predictions.geojson',
    clip: true
  }
};

export function getBoothAssets(assemblyConstituency, boothNumber) {
  const key = `${String(assemblyConstituency || '').toUpperCase().trim()}|${boothNumber}`;
  return BOOTH_ASSETS[key] || null;
}
