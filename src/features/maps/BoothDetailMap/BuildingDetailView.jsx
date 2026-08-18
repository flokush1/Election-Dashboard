import React from 'react';
import BuildingDetailPanel from '../../../components/BuildingDetailPanel.jsx';

const BuildingDetailView = ({ building, onClose }) => {
  if (!building) return null;
  return (
    <div className="fixed inset-0 z-[2000] bg-black/40 flex justify-end">
      <div className="h-full overflow-y-auto">
        <BuildingDetailPanel building={building} onClose={onClose} />
      </div>
    </div>
  );
};

export default BuildingDetailView;
