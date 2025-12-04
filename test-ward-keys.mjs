import { processElectoralData } from './src/shared/dataProcessor.js';
import electoralDataJson from './public/data/electoral-data.json' assert { type: 'json' };

const processed = processElectoralData(electoralDataJson);

console.log('='.repeat(80));
console.log('WARD KEYS IN PROCESSED DATA');
console.log('='.repeat(80));

const wardKeys = Object.keys(processed.wards);
console.log(`\nTotal wards: ${wardKeys.length}`);

console.log('\nR K Puram assembly wards:');
wardKeys
  .filter(key => {
    const ward = processed.wards[key];
    return ward.assembly === 'R K Puram';
  })
  .forEach(key => {
    const ward = processed.wards[key];
    console.log(`  Key: "${key}"`);
    console.log(`    Assembly: ${ward.assembly}`);
    console.log(`    Booths Won:`, ward.boothsWon);
  });

console.log('\n' + '='.repeat(80));
console.log('The GeoJSON has "R.K. PURAM" but ward key is probably "RK Puram"');
console.log('='.repeat(80));
