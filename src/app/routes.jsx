import React, { lazy } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import DashboardLayout from './DashboardLayout.jsx';

const ParliamentPage = lazy(() => import('../features/hierarchy/pages/ParliamentPage.jsx'));
const AssemblyPage = lazy(() => import('../features/hierarchy/pages/AssemblyPage.jsx'));
const WardPage = lazy(() => import('../features/hierarchy/pages/WardPage.jsx'));
const BoothPage = lazy(() => import('../features/booth/pages/BoothPage.jsx'));
const VoterPredictionPage = lazy(() => import('../features/predictions/pages/VoterPredictionPage.jsx'));

function AppRoutes() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/" element={<ParliamentPage />} />
        <Route path="/assembly/:assembly" element={<AssemblyPage />} />
        <Route path="/assembly/:assembly/ward/:ward" element={<WardPage />} />
        <Route path="/assembly/:assembly/ward/:ward/booth/:booth" element={<BoothPage />} />
        <Route path="/predictions" element={<VoterPredictionPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export default AppRoutes;
