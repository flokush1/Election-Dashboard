import { describe, expect, it } from 'vitest';
import {
  canonicalWardKey,
  formatNumber,
  formatPercentage,
  formatPreviewValue,
  getPartyColor,
  normalizeWardDisplay
} from '../shared/utils.js';

describe('canonicalWardKey', () => {
  it('normalizes punctuation, spacing, and aliases', () => {
    expect(canonicalWardKey('R.K. Puram')).toBe(canonicalWardKey('R K PURAM'));
    expect(canonicalWardKey('Chitaranjan Pk')).toBe(canonicalWardKey('Chittaranjan Park'));
    expect(canonicalWardKey('Hauz Khaz')).toBe(canonicalWardKey('Hauz Khas'));
    expect(canonicalWardKey('Ranjeet Nagar')).toBe(canonicalWardKey('Ranjit Nagar'));
  });

  it('returns empty string for falsy names', () => {
    expect(canonicalWardKey('')).toBe('');
    expect(canonicalWardKey(null)).toBe('');
  });
});

describe('format helpers', () => {
  it('formats compact numbers and percentages', () => {
    expect(formatNumber(1500)).toBe('1.5K');
    expect(formatPercentage(25, 100)).toBe('25.0%');
    expect(formatPreviewValue(0.42, 'turnout_ratio')).toBe('42.0%');
  });

  it('maps known party colors', () => {
    expect(getPartyColor('BJP')).toBe('#FF9933');
    expect(getPartyColor('Unknown')).toBe('#6B7280');
  });

  it('preserves short abbreviations in display names', () => {
    expect(normalizeWardDisplay('RK PURAM')).toBe('RK Puram');
  });
});
