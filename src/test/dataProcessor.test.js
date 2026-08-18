import { describe, expect, it } from 'vitest';
import { processElectoralData, processGeoJsonData } from '../shared/dataProcessor.js';

const booths = [
  {
    AssemblyName: 'NEW DELHI',
    AssemblyNo: 40,
    'Ward Name': 'Ward A',
    'Ward No.': 1,
    PartNo: 1,
    Winner: 'BJP',
    Total_Polled: 100,
    TotalPop: 200,
    BJP_Ratio: 0.5,
    AAP_Ratio: 0.3,
    Congress_Ratio: 0.1,
    Others_Ratio: 0.05,
    NOTA_Ratio: 0.05,
    Margin: 20,
    economic_category: 'MIDDLE CLASS'
  },
  {
    AssemblyName: 'NEW DELHI',
    AssemblyNo: 40,
    'Ward Name': 'Ward A',
    'Ward No.': 1,
    PartNo: 2,
    Winner: 'AAP',
    Total_Polled: 100,
    TotalPop: 200,
    BJP_Ratio: 0.2,
    AAP_Ratio: 0.6,
    Congress_Ratio: 0.1,
    Others_Ratio: 0.05,
    NOTA_Ratio: 0.05,
    Margin: 40,
    economic_category: 'PREMIUM AREAS'
  }
];

describe('processElectoralData', () => {
  it('aggregates parliament, assembly, and ward totals', () => {
    const processed = processElectoralData(booths);
    expect(processed.parliament.totalVotes).toBe(200);
    expect(processed.parliament.boothsWon.BJP).toBe(1);
    expect(processed.parliament.boothsWon.AAP).toBe(1);
    expect(processed.parliament.partyVotes.BJP).toBe(70);
    expect(processed.assemblies['NEW DELHI'].totalBooths).toBe(2);
    expect(processed.wards['Ward A'].assembly).toBe('NEW DELHI');
    expect(processed.booths).toHaveLength(2);
  });

  it('falls back to mock data for empty input', () => {
    const processed = processElectoralData([]);
    expect(processed.parliament.name).toBe('NEW DELHI');
    expect(processed.booths).toEqual([]);
  });
});

describe('processGeoJsonData', () => {
  it('returns FeatureCollections unchanged', () => {
    const geo = { type: 'FeatureCollection', features: [{ type: 'Feature' }] };
    expect(processGeoJsonData(geo, 'assembly')).toBe(geo);
  });
});
