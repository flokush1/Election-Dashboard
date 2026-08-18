import { useEffect, useState } from 'react';
import { processElectoralData, processGeoJsonData } from '../../../entities/electoral/processElectoralData.js';

export function useElectoralBootstrap() {
  const [data, setData] = useState(null);
  const [geoData, setGeoData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const loadGeoData = async (level, url) => {
      try {
        const response = await fetch(url);
        if (!response.ok) return [level, []];
        const geoJson = await response.json();
        return [level, processGeoJsonData(geoJson, level)];
      } catch {
        return [level, []];
      }
    };

    const loadData = async () => {
      try {
        setLoading(true);
        const [electoralData, assemblyGeo, parliamentGeo] = await Promise.all([
          fetch('/data/electoral-data.json').then((response) => response.ok ? response.json() : []),
          loadGeoData('assembly', '/data/assembly-boundaries.geojson'),
          loadGeoData('parliament', '/data/parliament-boundaries.geojson')
        ]);
        if (cancelled) return;
        setData(processElectoralData(electoralData));
        setGeoData(Object.fromEntries([assemblyGeo, parliamentGeo]));
        setLoading(false);
        Promise.all([
          loadGeoData('ward', '/data/ward-boundaries.geojson'),
          loadGeoData('booth', '/data/geospatial/New_Delhi_Booth_Data.geojson')
        ]).then((entries) => {
          if (!cancelled) {
            setGeoData((current) => ({ ...current, ...Object.fromEntries(entries) }));
          }
        });
      } catch (err) {
        if (cancelled) return;
        setError(err.message);
        setData(processElectoralData([]));
        setLoading(false);
      }
    };

    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, geoData, loading, error };
}
