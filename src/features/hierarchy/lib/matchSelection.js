import { canonicalWardKey } from '../../../shared/utils.js';

export const matchNamedKey = (name, dataObj = {}) => {
  if (!name || !dataObj) return null;
  if (dataObj[name]) return name;

  const exactInsensitive = Object.keys(dataObj).find(
    (key) => key.toLowerCase() === String(name).toLowerCase()
  );
  if (exactInsensitive) return exactInsensitive;

  const target = canonicalWardKey(name);
  return Object.keys(dataObj).find((key) => canonicalWardKey(key) === target) || null;
};

export const findBoothInWard = (booths = [], boothNumber, wardName) => {
  if (!Array.isArray(booths)) return null;
  return booths.find((booth) =>
    Number(booth.PartNo) === Number(boothNumber) &&
    (!wardName || booth['Ward Name'] === wardName)
  ) || null;
};

export const listAssemblies = (data) => (
  data?.assemblies ? Object.keys(data.assemblies).sort() : []
);

export const listWardsForAssembly = (data, assemblyName) => {
  if (!data?.wards || !assemblyName) return [];
  return Object.keys(data.wards)
    .filter((wardName) => data.wards[wardName].assembly === assemblyName)
    .sort();
};

export const listBoothsForWard = (data, wardName) => {
  if (!data?.booths || !wardName) return [];
  return data.booths
    .filter((booth) => booth['Ward Name'] === wardName)
    .sort((a, b) => a.PartNo - b.PartNo);
};
