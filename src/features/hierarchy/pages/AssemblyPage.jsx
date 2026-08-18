import React, { lazy } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useHierarchyNav } from '../hooks/useHierarchyNav.js';

const AssemblyLevel = lazy(() => import('../../../components/levels/AssemblyLevel.jsx'));

const AssemblyPage = () => {
  const { data, geoData } = useOutletContext();
  const nav = useHierarchyNav(data);
  if (!nav.selectedAssembly) return null;
  return (
    <AssemblyLevel
      data={data.assemblies[nav.selectedAssembly]}
      wards={data.wards}
      geoData={geoData}
      onNavigateToWard={nav.navigateToWard}
      onNavigateBack={nav.navigateBack}
      onNavigateHome={nav.navigateHome}
      onNavigateToVoterPrediction={nav.navigateToVoterPrediction}
      availableWards={nav.availableWards}
      selectedAssembly={nav.selectedAssembly}
      availableAssemblies={nav.availableAssemblies}
      onAssemblyChange={nav.navigateToAssembly}
    />
  );
};

export default AssemblyPage;
