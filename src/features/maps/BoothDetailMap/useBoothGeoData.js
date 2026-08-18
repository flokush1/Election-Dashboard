import { useEffect, useState } from 'react';
import { getBoothAssets } from '../config/boothAssets.js';
import { clipPlotsToBoundary, filterBoothFeatures } from './clipPlots.js';

export function useBoothGeoData(assemblyConstituency, boothNumber, hasDetailedData) {
  const [boothBoundaryData, setBoothBoundaryData] = useState(null);
  const [buildingData, setBuildingData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadGeoData = async () => {
      try {
        setLoading(true);
        setError(null);
        if (!hasDetailedData) {
          setBoothBoundaryData(null);
          setBuildingData(null);
          return;
        }
        const assets = getBoothAssets(assemblyConstituency, boothNumber);
        if (!assets) return;
        const [boundaryResp, buildingsResp] = await Promise.all([
          fetch(assets.boundaryUrl),
          fetch(assets.buildingsUrl)
        ]);
        let boundary = boundaryResp.ok ? await boundaryResp.json() : null;
        let buildings = buildingsResp.ok ? await buildingsResp.json() : null;
        if (boundary && !assets.clip && !assets.dedicated) {
          boundary = filterBoothFeatures(boundary, boothNumber, assemblyConstituency);
        }
        if (buildings && assets.clip && boundary) {
          buildings = clipPlotsToBoundary(buildings, boundary);
        }
        setBoothBoundaryData(boundary);
        setBuildingData(buildings);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadGeoData();
  }, [assemblyConstituency, boothNumber, hasDetailedData]);

  return { boothBoundaryData, buildingData, loading, error };
}
