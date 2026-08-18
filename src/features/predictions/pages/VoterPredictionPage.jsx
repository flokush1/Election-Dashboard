import React, { lazy } from 'react';
import { useOutletContext } from 'react-router-dom';
import { useHierarchyNav } from '../../hierarchy/hooks/useHierarchyNav.js';

const VoterPredictionPanel = lazy(() => import('../../../components/levels/VoterPredictionPanel.jsx'));

const VoterPredictionPage = () => {
  const { data } = useOutletContext();
  const nav = useHierarchyNav(data);
  return (
    <VoterPredictionPanel
      onNavigateBack={nav.navigateBack}
      onNavigateHome={nav.navigateHome}
    />
  );
};

export default VoterPredictionPage;
