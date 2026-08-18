import React, { lazy } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useHierarchyNav } from '../hooks/useHierarchyNav.js';

const ParliamentLevel = lazy(() => import('../../../components/levels/ParliamentLevel.jsx'));

const ParliamentPage = () => {
  const { data, geoData } = useOutletContext();
  const nav = useHierarchyNav(data);
  return (
    <ParliamentLevel
      data={data.parliament}
      assemblies={data.assemblies}
      geoData={geoData}
      onNavigateToAssembly={nav.navigateToAssembly}
      onNavigateToVoterPrediction={nav.navigateToVoterPrediction}
      availableAssemblies={nav.availableAssemblies}
    />
  );
};

export default ParliamentPage;
