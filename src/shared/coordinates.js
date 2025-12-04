// Booth coordinate mappings for New Delhi constituency
// Based on the GeoJSON data provided

export const BOOTH_COORDINATES = {
  "NEW DELHI": {
    "1": [28.6411507, 77.205307], // From the provided geospatial data
    "103": [28.58278, 77.21646], // Block-E, B.K.Dutt Colony - Booth 103
    // Add more booth coordinates as they become available
    "2": [28.6420, 77.2060],
    "3": [28.6430, 77.2070],
    "4": [28.6440, 77.2080],
    "5": [28.6450, 77.2090],
    // Default coordinates for New Delhi constituency
    default: [28.6139, 77.2090]
  },
  "R K PURAM": {
    "17": [28.577, 77.169], // Shanti Niketan area - RK Puram Booth 17
    default: [28.5629, 77.1824]
  }
};

// Booth metadata including polling station details
export const BOOTH_METADATA = {
  "NEW DELHI": {
    "1": {
      name: "ST THOMAS SCHOOL",
      address: "Mandir Marg",
      ward: "New Delhi",
      locality: "MANDIR MARG"
    },
    "103": {
      name: "Polling Station Booth 103",
      address: "Block-E, B.K.Dutt Colony",
      ward: "New Delhi",
      locality: "B.K. DUTT COLONY"
    }
  },
  "R K PURAM": {
    "17": {
      name: "MOUNT CARMEL SCHOOL ANAND NIKETAN",
      address: "A-21, West End Colony, Block D 3, Moti Bagh, New Delhi, Delhi 110021",
      ward: "RK Puram",
      locality: "ANAND NIKETAN"
    }
  }
};

// Function to get booth coordinates
export const getBoothCoordinates = (assemblyConstituency, boothNumber) => {
  const constituency = BOOTH_COORDINATES[assemblyConstituency];
  if (!constituency) {
    return BOOTH_COORDINATES["NEW DELHI"].default;
  }
  
  return constituency[boothNumber?.toString()] || constituency.default;
};

// Function to get booth metadata
export const getBoothMetadata = (assemblyConstituency, boothNumber) => {
  const normalizedAssembly = assemblyConstituency?.toUpperCase().trim();
  const boothStr = boothNumber?.toString();
  
  const metadata = BOOTH_METADATA[normalizedAssembly];
  if (!metadata) {
    return null;
  }
  
  return metadata[boothStr] || null;
};

// Function to check if detailed geospatial data is available for a booth
export const hasDetailedBoothData = (assemblyConstituency, boothNumber) => {
  console.log('hasDetailedBoothData check:', assemblyConstituency, boothNumber);
  
  // Currently detailed data available for:
  // - NEW DELHI Booth 1 (ST THOMAS SCHOOL - MANDIR MARG)
  // - R K PURAM Booth 17 (Shanti Niketan plots inside RK Puram booth boundary)
  // Handle mixed casing in assembly names
  const normalizedAssembly = assemblyConstituency?.toUpperCase().trim();
  const boothStr = boothNumber?.toString();

  const result = (
    (normalizedAssembly === "NEW DELHI" && boothStr === "1") ||
    (normalizedAssembly === "NEW DELHI" && boothStr === "103") ||
    (normalizedAssembly === "R K PURAM" && boothStr === "17")
  );

  console.log('hasDetailedBoothData result:', result, 'normalized assembly:', normalizedAssembly);
  return result;
};

// Assembly constituency center coordinates
export const ASSEMBLY_CENTERS = {
  "NEW DELHI": [28.6139, 77.2090],
  "KAROL BAGH": [28.6519, 77.1909],
  "PATEL NAGAR": [28.6562, 77.1738],
  "MOTI NAGAR": [28.6606, 77.1462],
  "MADIPUR": [28.6714, 77.1321],
  "RAJINDER NAGAR": [28.6417, 77.1875],
  "JANGPURA": [28.5766, 77.2436],
  "KASTURBA NAGAR": [28.5958, 77.2319],
  "MALVIYA NAGAR": [28.5267, 77.2056],
  "R K PURAM": [28.5629, 77.1824],
  "GREATER KAILASH": [28.5494, 77.2425],
  "DELHI CANTT": [28.5797, 77.1025],
  "RAJOURI GARDEN": [28.6417, 77.1244],
  "NERELA": [28.8456, 77.0921] // Added missing assembly
};

// Get center coordinates for an assembly constituency
export const getAssemblyCenter = (assemblyName) => {
  return ASSEMBLY_CENTERS[assemblyName] || ASSEMBLY_CENTERS["NEW DELHI"];
};