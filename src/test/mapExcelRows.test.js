import { describe, expect, it } from 'vitest';
import { mapExcelRows } from '../features/predictions/lib/mapExcelRows.js';

describe('mapExcelRows', () => {
  it('maps spreadsheet rows onto canonical voter fields', () => {
    const mapped = mapExcelRows([{ 'voters id': 'AB123', name: 'Ada', age: 33, Locality: 'Madipur' }]);
    expect(mapped[0].voter_id).toBe('AB123');
    expect(mapped[0].name).toBe('Ada');
    expect(mapped[0].locality).toBe('MADIPUR');
  });
});
