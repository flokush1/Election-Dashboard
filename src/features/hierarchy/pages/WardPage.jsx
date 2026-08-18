import React, { lazy } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useHierarchyNav } from '../hooks/useHierarchyNav.js';

const WardLevel = lazy(() => import('../../../components/levels/WardLevel.jsx'));

const WardPage = () => {
  const { data, geoData } = useOutletContext();
  const nav = useHierarchyNav(data);
  if (!nav.selectedWard) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <h2 className="text-xl font-semibold mb-4">Ward Not Found</h2>
          <button onClick={nav.navigateBack} className="px-4 py-2 bg-blue-600 text-white rounded-lg">Go Back</button>
        </div>
      </div>
    );
  }
  return (
    <WardLevel
      data={data.wards[nav.selectedWard]}
      booths={data.booths}
      geoData={geoData}
      onNavigateToBooth={nav.navigateToBooth}
      onNavigateBack={nav.navigateBack}
      onNavigateHome={nav.navigateHome}
      availableBooths={nav.availableBooths}
      selectedWard={nav.selectedWard}
      availableWards={nav.availableWards}
      onWardChange={nav.navigateToWard}
    />
  );
};

export default WardPage;
