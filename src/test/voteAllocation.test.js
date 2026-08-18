import { describe, expect, it } from 'vitest';
import { allocateBoothVotes, allocateVotes } from '../features/booth/voteAllocation.js';

describe('vote allocation', () => {
  it('uses largest remainder so allocated votes equal the total', () => {
    const votes = allocateVotes({ BJP: 0.333, AAP: 0.333, Congress: 0.334 }, 100);
    expect(Object.values(votes).reduce((sum, value) => sum + value, 0)).toBe(100);
  });

  it('derives winner and margin from allocated votes', () => {
    const result = allocateBoothVotes(
      { BJP: 0.51, AAP: 0.29, Congress: 0.1, Others: 0.05, NOTA: 0.05 },
      200
    );
    expect(result.winnerParty).toBe('BJP');
    expect(Object.values(result.partyVotes).reduce((sum, value) => sum + value, 0)).toBe(200);
    expect(result.recomputedMargin).toBeGreaterThan(0);
  });
});
