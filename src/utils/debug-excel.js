// Debug script to test Excel parsing
import * as XLSX from 'xlsx';

// Test data loading function
export const testExcelParsing = async (file) => {
  console.log('=== DEBUGGING EXCEL PARSING ===');
  console.log('File name:', file.name);
  console.log('File size:', (file.size / (1024 * 1024)).toFixed(2), 'MB');
  console.log('File type:', file.type);
  
  try {
    // Read file
    console.log('1. Reading file...');
    const arrayBuffer = await file.arrayBuffer();
    console.log('   File read successfully, size:', arrayBuffer.byteLength);
    
    // Parse workbook
    console.log('2. Parsing workbook...');
    const workbook = XLSX.read(arrayBuffer, { 
      type: 'array',
      cellDates: false,
      cellNF: false,
      cellText: true,
      raw: false
    });
    
    console.log('   Workbook parsed successfully');
    console.log('   Sheet names:', workbook.SheetNames);
    
    // Get first worksheet
    console.log('3. Processing worksheet...');
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    
    if (!worksheet) {
      throw new Error('No worksheet found');
    }
    
    // Get sheet info
    const range = XLSX.utils.decode_range(worksheet['!ref']);
    console.log('   Sheet range:', worksheet['!ref']);
    console.log('   Estimated rows:', range.e.r + 1);
    console.log('   Estimated cols:', range.e.c + 1);
    
    // Convert small sample to JSON
    console.log('4. Converting sample to JSON...');
    const sampleData = XLSX.utils.sheet_to_json(worksheet, {
      header: 1,
      range: 0, // Start from beginning
      defval: '',
      blankrows: false,
      raw: false
    });
    
    console.log('   Sample data length:', sampleData.length);
    console.log('   First 3 rows:', sampleData.slice(0, 3));
    
    // Validate headers
    if (sampleData.length > 0) {
      const headers = sampleData[0];
      console.log('   Headers:', headers);
      console.log('   Header count:', headers.length);
      
      // Check for voter ID column
      const voterIdColumns = headers.filter(h => 
        h && h.toString().toLowerCase().includes('voter') || 
        h.toString().toLowerCase().includes('id') ||
        h.toString().toLowerCase().includes('epic')
      );
      console.log('   Potential voter ID columns:', voterIdColumns);
    }
    
    console.log('=== PARSING SUCCESSFUL ===');
    return {
      success: true,
      sheetNames: workbook.SheetNames,
      rowCount: range.e.r + 1,
      colCount: range.e.c + 1,
      headers: sampleData.length > 0 ? sampleData[0] : [],
      sampleRows: sampleData.slice(1, 4) // First 3 data rows
    };
    
  } catch (error) {
    console.error('=== PARSING FAILED ===');
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
    return {
      success: false,
      error: error.message
    };
  }
};

// Test function to call from console
window.testExcelUpload = testExcelParsing;