// Enhanced data processor for real electoral data
export const processElectoralData = (rawData) => {
  if (!rawData || !Array.isArray(rawData) || rawData.length === 0) {
    console.log('No raw data available, using mock data');
    return getMockData();
  }

  console.log(`Processing ${rawData.length} booth records`);
  
  const processed = {
    parliament: {},
    assemblies: {},
    wards: {},
    booths: rawData,
    summary: {}
  };

  // Parliament level aggregation
  const totalVotes = rawData.reduce((sum, booth) => sum + (booth.Total_Polled || 0), 0);
  const totalPopulation = rawData.reduce((sum, booth) => sum + (booth.TotalPop || 0), 0);
  
  const partyVotes = {
    BJP: Math.round(rawData.reduce((sum, booth) => sum + ((booth.BJP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
    AAP: Math.round(rawData.reduce((sum, booth) => sum + ((booth.AAP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
    Congress: Math.round(rawData.reduce((sum, booth) => sum + ((booth.Congress_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
    Others: Math.round(rawData.reduce((sum, booth) => sum + ((booth.Others_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
    NOTA: Math.round(rawData.reduce((sum, booth) => sum + ((booth.NOTA_Ratio || 0) * (booth.Total_Polled || 0)), 0))
  };

  processed.parliament = {
    name: 'NEW DELHI',
    totalVotes,
    totalPopulation,
    partyVotes,
    totalBooths: rawData.length,
    boothsWon: {
      BJP: rawData.filter(b => b.Winner === 'BJP').length,
      AAP: rawData.filter(b => b.Winner === 'AAP').length,
      Congress: rawData.filter(b => b.Winner === 'Congress').length,
      Others: rawData.filter(b => b.Winner === 'Others').length,
      Tie: rawData.filter(b => b.Winner === 'Tie').length
    },
    averageMargin: calculateWeightedMargin(rawData),
    demographics: calculateDemographics(rawData)
  };

  // Assembly level aggregation
  const assemblyGroups = groupBy(rawData, 'AssemblyName');
  Object.keys(assemblyGroups).forEach(assemblyName => {
    const booths = assemblyGroups[assemblyName];
    const assemblyTotalVotes = booths.reduce((sum, booth) => sum + (booth.Total_Polled || 0), 0);
    const assemblyTotalPop = booths.reduce((sum, booth) => sum + (booth.TotalPop || 0), 0);
    
    processed.assemblies[assemblyName] = {
      name: assemblyName,
      number: booths[0].AssemblyNo,
      totalVotes: assemblyTotalVotes,
      totalPopulation: assemblyTotalPop,
      totalBooths: booths.length,
      partyVotes: {
        BJP: Math.round(booths.reduce((sum, booth) => sum + ((booth.BJP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        AAP: Math.round(booths.reduce((sum, booth) => sum + ((booth.AAP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        Congress: Math.round(booths.reduce((sum, booth) => sum + ((booth.Congress_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        Others: Math.round(booths.reduce((sum, booth) => sum + ((booth.Others_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        NOTA: Math.round(booths.reduce((sum, booth) => sum + ((booth.NOTA_Ratio || 0) * (booth.Total_Polled || 0)), 0))
      },
      boothsWon: {
        BJP: booths.filter(b => b.Winner === 'BJP').length,
        AAP: booths.filter(b => b.Winner === 'AAP').length,
        Congress: booths.filter(b => b.Winner === 'Congress').length,
        Others: booths.filter(b => b.Winner === 'Others').length,
        Tie: booths.filter(b => b.Winner === 'Tie').length
      },
      averageMargin: calculateWeightedMargin(booths),
      demographics: calculateDemographics(booths),
      economics: calculateEconomics(booths)
    };
  });

  // Ward level aggregation
  const wardGroups = groupBy(rawData.filter(b => b['Ward Name']), 'Ward Name');
  Object.keys(wardGroups).forEach(wardName => {
    const booths = wardGroups[wardName];
    const wardTotalVotes = booths.reduce((sum, booth) => sum + (booth.Total_Polled || 0), 0);
    const wardTotalPop = booths.reduce((sum, booth) => sum + (booth.TotalPop || 0), 0);
    
    processed.wards[wardName] = {
      name: wardName,
      number: booths[0]['Ward No.'],
      assembly: booths[0].AssemblyName,
      totalVotes: wardTotalVotes,
      totalPopulation: wardTotalPop,
      totalBooths: booths.length,
      partyVotes: {
        BJP: Math.round(booths.reduce((sum, booth) => sum + ((booth.BJP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        AAP: Math.round(booths.reduce((sum, booth) => sum + ((booth.AAP_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        Congress: Math.round(booths.reduce((sum, booth) => sum + ((booth.Congress_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        Others: Math.round(booths.reduce((sum, booth) => sum + ((booth.Others_Ratio || 0) * (booth.Total_Polled || 0)), 0)),
        NOTA: Math.round(booths.reduce((sum, booth) => sum + ((booth.NOTA_Ratio || 0) * (booth.Total_Polled || 0)), 0))
      },
      boothsWon: {
        BJP: booths.filter(b => b.Winner === 'BJP').length,
        AAP: booths.filter(b => b.Winner === 'AAP').length,
        Congress: booths.filter(b => b.Winner === 'Congress').length,
        Others: booths.filter(b => b.Winner === 'Others').length,
        Tie: booths.filter(b => b.Winner === 'Tie').length
      },
      averageMargin: calculateWeightedMargin(booths),
      demographics: calculateDemographics(booths),
      economics: calculateEconomics(booths)
    };
  });

  console.log(`Processed: ${Object.keys(processed.assemblies).length} assemblies, ${Object.keys(processed.wards).length} wards`);
  return processed;
};

const groupBy = (array, key) => {
  return array.reduce((result, currentValue) => {
    const group = currentValue[key];
    if (group) {
      (result[group] = result[group] || []).push(currentValue);
    }
    return result;
  }, {});
};

const calculateWeightedMargin = (booths) => {
  if (!booths || booths.length === 0) return 0;
  
  const totalVotes = booths.reduce((sum, booth) => sum + (booth.Total_Polled || 0), 0);
  if (totalVotes === 0) return 0;
  
  // Weight margins by vote count for more accurate representation
  return booths.reduce((sum, booth) => {
    const votes = booth.Total_Polled || 0;
    const margin = booth.Margin || 0;
    return sum + (margin * votes);
  }, 0) / totalVotes;
};

const calculateDemographics = (booths) => {
  if (!booths || booths.length === 0) return {};

  const totalPop = booths.reduce((sum, booth) => sum + (booth.TotalPop || 0), 0);
  
  if (totalPop === 0) return {};

  // Weighted average calculations for demographics
  const calculateWeightedAverage = (ratioField) => {
    return booths.reduce((sum, booth) => {
      const population = booth.TotalPop || 0;
      const ratio = booth[ratioField] || 0;
      return sum + (ratio * population);
    }, 0) / totalPop;
  };

  return {
    totalPopulation: totalPop,
    averagePopulation: Math.round(totalPop / booths.length),
    ageGroups: {
      '18-25': calculateWeightedAverage('Age_18-25_Ratio'),
      '26-35': calculateWeightedAverage('Age_26-35_Ratio'),
      '36-45': calculateWeightedAverage('Age_36-45_Ratio'),
      '46-60': calculateWeightedAverage('Age_46-60_Ratio'),
      '60+': calculateWeightedAverage('Age_60+_Ratio')
    },
    genderRatio: {
      male: calculateWeightedAverage('Male_Ratio'),
      female: calculateWeightedAverage('Female_Ratio')
    },
    religion: {
      hindu: calculateWeightedAverage('Religion_Hindu_Ratio'),
      muslim: calculateWeightedAverage('Religion_Muslim_Ratio'),
      sikh: calculateWeightedAverage('Religion_Sikh_Ratio'),
      christian: calculateWeightedAverage('Religion_Christian_Ratio'),
      buddhist: calculateWeightedAverage('Religion_Buddhist_Ratio'),
      jain: calculateWeightedAverage('Religion_Jain_Ratio'),
      other: calculateWeightedAverage('Religion_Unknown_Ratio')
    },
    caste: {
      sc: calculateWeightedAverage('Caste_Sc_Ratio'),
      obc: calculateWeightedAverage('Caste_Obc_Ratio'),
      brahmin: calculateWeightedAverage('Caste_Brahmin_Ratio'),
      kshatriya: calculateWeightedAverage('Caste_Kshatriya_Ratio'),
      vaishya: calculateWeightedAverage('Caste_Vaishya_Ratio'),
      st: calculateWeightedAverage('Caste_St_Ratio')
    }
  };
};

const calculateEconomics = (booths) => {
  if (!booths || booths.length === 0) return {};

  return {
    categories: {
      'LOW INCOME AREAS': booths.filter(b => b.economic_category === 'LOW INCOME AREAS').length,
      'LOWER MIDDLE CLASS': booths.filter(b => b.economic_category === 'LOWER MIDDLE CLASS').length,
      'MIDDLE CLASS': booths.filter(b => b.economic_category === 'MIDDLE CLASS').length,
      'UPPER MIDDLE CLASS': booths.filter(b => b.economic_category === 'UPPER MIDDLE CLASS').length,
      'PREMIUM AREAS': booths.filter(b => b.economic_category === 'PREMIUM AREAS').length
    },
    averageLandRate: booths.reduce((sum, booth) => sum + (booth.land_rate_per_sqm || 0), 0) / booths.length,
    averageConstructionCost: booths.reduce((sum, booth) => sum + (booth.construction_cost_per_sqm || 0), 0) / booths.length
  };
};

const getMockData = () => {
  return {
    parliament: {
      name: 'NEW DELHI',
      totalVotes: 880747,
      totalPopulation: 1538430,
      partyVotes: { BJP: 330000, AAP: 470000, Congress: 60000, Others: 15000, NOTA: 5747 },
      totalBooths: 1395,
      boothsWon: { BJP: 819, AAP: 542, Congress: 32, Others: 2, Tie: 0 },
      averageMargin: 124.9,
      demographics: {
        totalPopulation: 1538430,
        averagePopulation: 1102,
        ageGroups: { '18-25': 0.15, '26-35': 0.25, '36-45': 0.22, '46-60': 0.23, '60+': 0.15 },
        genderRatio: { male: 0.54, female: 0.46 },
        religion: { hindu: 0.78, muslim: 0.12, sikh: 0.06, christian: 0.02, buddhist: 0.01, jain: 0.01, other: 0.0 }
      }
    },
    assemblies: {},
    wards: {},
    booths: []
  };
};

export const processGeoJsonData = (geoJsonData, level) => {
  if (!geoJsonData || !geoJsonData.features) return geoJsonData || [];
  
  // Return the geoJsonData as-is since InteractiveMap expects full GeoJSON
  return geoJsonData;
};

const getNameField = (level) => {
  switch(level) {
    case 'parliament': return 'PC_Name';
    case 'assembly': return 'A_CNST_NM';
    case 'ward': return 'WardName';
    default: return 'name';
  }
};

const getNumberField = (level) => {
  switch(level) {
    case 'parliament': return 'PC_No';
    case 'assembly': return 'AC_No';
    case 'ward': return 'Ward_No';
    default: return 'number';
  }
};