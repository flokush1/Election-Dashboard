import { describe, expect, it } from 'vitest';
import {
  findBoothInWard,
  listAssemblies,
  listBoothsForWard,
  listWardsForAssembly,
  matchNamedKey
} from '../features/hierarchy/lib/matchSelection.js';

const data = {
  assemblies: { 'NEW DELHI': {}, 'R K PURAM': {} },
  wards: {
    'Ward A': { assembly: 'NEW DELHI' },
    'Ward B': { assembly: 'R K PURAM' }
  },
  booths: [
    { PartNo: 17, 'Ward Name': 'Ward B' },
    { PartNo: 1, 'Ward Name': 'Ward A' }
  ]
};

describe('hierarchy selection', () => {
  it('matches assembly names with punctuation aliases', () => {
    expect(matchNamedKey('R.K. Puram', data.assemblies)).toBe('R K PURAM');
    expect(matchNamedKey('new delhi', data.assemblies)).toBe('NEW DELHI');
  });

  it('lists wards and booths for the current selection', () => {
    expect(listAssemblies(data)).toEqual(['NEW DELHI', 'R K PURAM']);
    expect(listWardsForAssembly(data, 'NEW DELHI')).toEqual(['Ward A']);
    expect(listBoothsForWard(data, 'Ward B')[0].PartNo).toBe(17);
    expect(findBoothInWard(data.booths, 1, 'Ward A').PartNo).toBe(1);
  });
});
