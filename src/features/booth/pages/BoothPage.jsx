import React, { lazy } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useHierarchyNav } from '../../hierarchy/hooks/useHierarchyNav.js';

const BoothLevel = lazy(() => import('../../../components/levels/BoothLevel.jsx'));

const BoothPage = () => {
  const { data, geoData } = useOutletContext();
  const nav = useHierarchyNav(data);
  return (
    <BoothLevel
      data={nav.selectedBoothData}
      geoData={geoData}
      onNavigateBack={nav.navigateBack}
      onNavigateHome={nav.navigateHome}
      availableBooths={nav.availableBooths}
      selectedBooth={nav.selectedBooth}
      onBoothChange={nav.navigateToBooth}
    />
  );
};

export default BoothPage;
