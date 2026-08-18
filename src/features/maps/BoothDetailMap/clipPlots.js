import * as turf from '@turf/turf';

export function clipPlotsToBoundary(buildingGeoData, boundaryCollection) {
  if (!buildingGeoData?.features || !boundaryCollection?.features?.length) {
    return buildingGeoData;
  }
  const boundaryGeom = boundaryCollection.features[0].geometry;
  const boundaryPoly = boundaryGeom.type === 'MultiPolygon'
    ? turf.multiPolygon(boundaryGeom.coordinates)
    : turf.polygon(boundaryGeom.coordinates);

  const clippedFeatures = [];
  for (const feat of buildingGeoData.features) {
    const geom = feat.geometry;
    if (!geom || geom.type !== 'Polygon') continue;
    try {
      const plotPoly = turf.polygon(geom.coordinates);
      if (turf.booleanIntersects(plotPoly, boundaryPoly)) {
        const clipped = turf.intersect(plotPoly, boundaryPoly);
        clippedFeatures.push({
          type: 'Feature',
          properties: feat.properties || {},
          geometry: clipped ? clipped.geometry : geom
        });
      }
    } catch {
      clippedFeatures.push(feat);
    }
  }
  return { ...buildingGeoData, features: clippedFeatures };
}

export function filterBoothFeatures(geoData, boothNumber, assemblyConstituency) {
  if (!geoData?.features) return geoData;
  return {
    ...geoData,
    features: geoData.features.filter((feature) => {
      const boothMatch = feature.properties.Booth_No?.toString() === boothNumber?.toString();
      const assemblyMatch = feature.properties.A_CNST_NM?.toUpperCase().trim() === assemblyConstituency?.toUpperCase().trim();
      return boothMatch && assemblyMatch;
    })
  };
}
