import { useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  findBoothInWard,
  listAssemblies,
  listBoothsForWard,
  listWardsForAssembly,
  matchNamedKey
} from '../lib/matchSelection.js';

const encode = (value) => encodeURIComponent(value || '');

export function useHierarchyNav(data) {
  const navigate = useNavigate();
  const params = useParams();

  const selectedAssembly = matchNamedKey(params.assembly ? decodeURIComponent(params.assembly) : null, data?.assemblies || {});
  const selectedWard = matchNamedKey(params.ward ? decodeURIComponent(params.ward) : null, data?.wards || {});
  const selectedBooth = params.booth ? decodeURIComponent(params.booth) : null;

  const availableAssemblies = useMemo(() => listAssemblies(data), [data]);
  const availableWards = useMemo(() => listWardsForAssembly(data, selectedAssembly), [data, selectedAssembly]);
  const availableBooths = useMemo(() => listBoothsForWard(data, selectedWard), [data, selectedWard]);
  const selectedBoothData = useMemo(
    () => findBoothInWard(data?.booths, selectedBooth, selectedWard),
    [data, selectedBooth, selectedWard]
  );

  return {
    selectedAssembly,
    selectedWard,
    selectedBooth,
    selectedBoothData,
    availableAssemblies,
    availableWards,
    availableBooths,
    navigateToAssembly: (assemblyName) => {
      const key = matchNamedKey(assemblyName, data?.assemblies || {});
      if (key) navigate(`/assembly/${encode(key)}`);
    },
    navigateToWard: (wardName) => {
      const key = matchNamedKey(wardName, data?.wards || {});
      if (key && selectedAssembly) navigate(`/assembly/${encode(selectedAssembly)}/ward/${encode(key)}`);
    },
    navigateToBooth: (boothNumber) => {
      if (selectedAssembly && selectedWard) {
        navigate(`/assembly/${encode(selectedAssembly)}/ward/${encode(selectedWard)}/booth/${encode(boothNumber)}`);
      }
    },
    navigateToVoterPrediction: () => navigate('/predictions'),
    navigateBack: () => navigate(-1),
    navigateHome: () => navigate('/')
  };
}
