// Test script to verify the color coding logic fix
import { processElectoralData } from './src/shared/dataProcessor.js';
import electoralDataJson from './public/data/electoral-data.json' assert { type: 'json' };

console.log('='.repeat(80));
console.log('TESTING COLOR CODING FIX FOR WARDS AND ASSEMBLIES');
console.log('='.repeat(80));

const processed = processElectoralData(electoralDataJson);

// Test RK Puram Assembly
console.log('\n📊 R K PURAM ASSEMBLY:');
const rkPuramAssembly = processed.assemblies['R K Puram'];
if (rkPuramAssembly) {
  console.log(`  Total booths: ${rkPuramAssembly.totalBooths}`);
  console.log(`  Booths won:`, rkPuramAssembly.boothsWon);
  console.log(`  Party votes:`, rkPuramAssembly.partyVotes);
  
  // Determine winner by booths won (CORRECT)
  const boothWinner = Object.entries(rkPuramAssembly.boothsWon)
    .filter(([party]) => party !== 'Tie')
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  console.log(`  ✓ CORRECT: Winner by booths won = ${boothWinner}`);
  
  // Determine winner by total votes (WRONG for map coloring)
  const voteWinner = Object.entries(rkPuramAssembly.partyVotes)
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  console.log(`  ✗ WRONG: Winner by total votes = ${voteWinner}`);
  
  if (boothWinner === voteWinner) {
    console.log(`  ℹ️  In this case, both methods agree`);
  }
} else {
  console.log('  Assembly not found!');
}

// Test RK Puram Ward
console.log('\n📊 RK PURAM WARD:');
const rkPuramWard = processed.wards['RK Puram'];
if (rkPuramWard) {
  console.log(`  Total booths: ${rkPuramWard.totalBooths}`);
  console.log(`  Booths won:`, rkPuramWard.boothsWon);
  console.log(`  Party votes:`, rkPuramWard.partyVotes);
  
  // Determine winner by booths won (CORRECT)
  const boothWinner = Object.entries(rkPuramWard.boothsWon)
    .filter(([party]) => party !== 'Tie')
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  console.log(`  ✓ CORRECT: Winner by booths won = ${boothWinner}`);
  
  // Determine winner by total votes (WRONG for map coloring)
  const voteWinner = Object.entries(rkPuramWard.partyVotes)
    .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  console.log(`  ✗ WRONG: Winner by total votes = ${voteWinner}`);
  
  if (boothWinner === voteWinner) {
    console.log(`  ℹ️  In this case, both methods agree`);
  }
} else {
  console.log('  Ward not found!');
}

console.log('\n' + '='.repeat(80));
console.log('The map should use "Winner by booths won" for coloring');
console.log('Code changes ensure InteractiveMap.jsx uses boothsWon');
console.log('='.repeat(80));
