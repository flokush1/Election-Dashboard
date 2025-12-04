// Consolidated utility functions for the electoral dashboard
export const formatNumber = (num) => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
};

export const formatPercentage = (value, total) => {
  if (!total || total === 0) return '0%';
  return ((value / total) * 100).toFixed(1) + '%';
};

// Format ratio columns as percentages for data preview
export const formatPreviewValue = (value, columnName) => {
  // List of columns that should be displayed as percentages
  const ratioColumns = [
    'male_female_ratio',
    'gender_ratio', 
    'turnout_ratio',
    'victory_margin_ratio',
    'vote_share'
  ];
  
  // Check if this column should be formatted as percentage
  const isRatioColumn = ratioColumns.some(ratioCol => 
    columnName.toLowerCase().includes(ratioCol.toLowerCase()) ||
    columnName.toLowerCase().includes('ratio') ||
    columnName.toLowerCase().includes('share') ||
    columnName.toLowerCase().includes('percentage')
  );
  
  if (isRatioColumn && value !== null && value !== undefined && value !== '') {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      // If value is between 0 and 1, treat as ratio and convert to percentage
      if (numValue >= 0 && numValue <= 1) {
        return (numValue * 100).toFixed(1) + '%';
      }
      // If value is greater than 1 but less than 100, likely already a percentage
      else if (numValue > 1 && numValue <= 100) {
        return numValue.toFixed(1) + '%';
      }
      // For values like male_female_ratio (e.g., 1.05 means 105 males per 100 females)
      else if (columnName.toLowerCase().includes('male_female_ratio') || 
               columnName.toLowerCase().includes('gender_ratio')) {
        return `${numValue.toFixed(2)}:1`;
      }
    }
  }
  
  return value;
};

export const getPartyColor = (party) => {
  const colors = {
    'BJP': '#FF9933',
    'AAP': '#0066CC', 
    'Congress': '#00CC66',
    'Others': '#6B7280',
    'NOTA': '#9CA3AF',
    // Gender colors (consistent with DemographicsChart)
    'Male': '#3B82F6',    // Blue
    'Female': '#EC4899'   // Pink
  };
  return colors[party] || '#6B7280';
};

export const getPartyColorClass = (party) => {
  const classes = {
    'BJP': 'text-orange-500 bg-orange-50 border-orange-200',
    'AAP': 'text-blue-600 bg-blue-50 border-blue-200',
    'Congress': 'text-green-600 bg-green-50 border-green-200',
    'Others': 'text-gray-600 bg-gray-50 border-gray-200',
    'NOTA': 'text-gray-500 bg-gray-50 border-gray-200',
    // Gender color classes (consistent with DemographicsChart)
    'Male': 'text-blue-600 bg-blue-50 border-blue-200',
    'Female': 'text-pink-600 bg-pink-50 border-pink-200'
  };
  return classes[party] || 'text-gray-600 bg-gray-50 border-gray-200';
};

export const calculateMarginCategory = (margin) => {
  if (margin < 50) return 'Very Close';
  if (margin < 100) return 'Close';
  if (margin < 200) return 'Comfortable';
  return 'Safe';
};

export const getMarginColor = (margin) => {
  if (margin < 50) return '#EF4444'; // red
  if (margin < 100) return '#F59E0B'; // yellow
  if (margin < 200) return '#10B981'; // green
  return '#3B82F6'; // blue
};

export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Basic class name utility (simplified version of clsx + twMerge)
export const cn = (...classes) => {
  return classes.filter(Boolean).join(' ');
};

// --- Name normalization helpers for robust GeoJSON ↔ Excel matching ---

// Title-case while preserving short all-caps abbreviations (e.g., RK, CR)
export const titleCasePreserveAbbrev = (name) => {
  if (!name) return '';
  return String(name)
    .split(' ')
    .filter(Boolean)
    .map((word) => {
      if (word === word.toUpperCase() && word.length <= 3) return word; // keep RK, CR
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(' ');
};

// Canonical key used for fuzzy matching: strip punctuation/spaces and alias known variants
export const canonicalWardKey = (name) => {
  if (!name) return '';
  let s = String(name)
    .replace(/\./g, ' ') // dots to space
    .replace(/[\-_]/g, ' ') // hyphen/underscore to space
    .replace(/\s+/g, ' ') // collapse spaces
    .trim()
    .toLowerCase();

  // Correct common spelling/alias variants BEFORE removing spaces
  s = s
    .replace(/\bchitaranjan\b/g, 'chittaranjan')
    .replace(/\bchitranjan\b/g, 'chittaranjan')
    .replace(/\bchittranjan\b/g, 'chittaranjan')
    .replace(/\bchittaranjan pk\b/g, 'chittaranjan park')
    .replace(/\b(c r park|cr park|c r p|c r)\b/g, 'chittaranjan park')
    // Common double-vowel variants in GeoJSON vs Excel
    .replace(/\branjeet\b/g, 'ranjit')
    .replace(/\bbaljeet\b/g, 'baljit')
    // Handle Hauz Khas vs Hauz Khaz misspelling
    .replace(/\bhauz\s+khaz\b/g, 'hauz khas');

  // Remove spaces to form the key
  return s.replace(/\s+/g, '');
};

// Produce a display-friendly normalized ward name (title case + abbrev preservation)
export const normalizeWardDisplay = (name) => {
  if (!name) return '';
  const noDots = String(name).replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
  return titleCasePreserveAbbrev(noDots);
};