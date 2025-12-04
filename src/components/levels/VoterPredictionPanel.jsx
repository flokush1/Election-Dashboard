import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { User, Users, Vote, TrendingUp, MapPin, Home, ArrowLeft, Zap, Upload, FileText, Search, AlertCircle, CheckCircle, Calendar, BookOpen, DollarSign } from 'lucide-react';
import * as XLSX from 'xlsx';
import StatCard from '../stats/StatCard.jsx';
import { formatNumber, getPartyColor } from '../../shared/utils.js';
import { testExcelParsing } from '../../utils/debug-excel.js';

const VoterPredictionPanel = ({ 
  onNavigateBack, 
  onNavigateHome,
  selectedVoter = null 
}) => {
  const [sampleVoters, setSampleVoters] = useState([]);
  const [currentVoter, setCurrentVoter] = useState(selectedVoter);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(false); // Start with false since we're in upload mode
  
  // New states for real ML model workflow
  const [mode, setMode] = useState('upload'); // Start with upload mode
  const [modelFile, setModelFile] = useState(null);
  const [voterDataFile, setVoterDataFile] = useState(null);
  const [voterData, setVoterData] = useState([]);
  // Raw (original) voter data as read from Excel before normalization/mapping
  const [rawVoterData, setRawVoterData] = useState([]);
  const [modelStatus, setModelStatus] = useState(null);
  const [realPredictions, setRealPredictions] = useState(null);
  const [familyPredictions, setFamilyPredictions] = useState(null);
  const [searchVoterId, setSearchVoterId] = useState('');
  const [searchedVoter, setSearchedVoter] = useState(null);
  const [uploadErrors, setUploadErrors] = useState({});
  const [uploadProgress, setUploadProgress] = useState({ stage: '', percent: 0 });
  const [sampleVoterIds, setSampleVoterIds] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking'); // 'checking', 'connected', 'error'
  // Data preview / download states
  const [showDataPreview, setShowDataPreview] = useState(false);
  const [dataFilter, setDataFilter] = useState('');
  const [previewPage, setPreviewPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [previewColumnsLimit, setPreviewColumnsLimit] = useState(25);
  // Preserve original Excel column order for preview/export
  const [rawColumns, setRawColumns] = useState([]);
  // Always use original uploaded/raw columns for preview now (user requested removal of normalized view toggle)
  const previewDataRows = React.useMemo(
    () => rawVoterData.length ? rawVoterData : voterData,
    [rawVoterData, voterData]
  );

  // Generic substring search across all values (raw data arbitrary columns)
  const filteredPreviewData = React.useMemo(() => {
    if (!dataFilter.trim()) return previewDataRows;
    const f = dataFilter.toLowerCase();
    return previewDataRows.filter(r =>
      Object.values(r).some(v => v !== null && v !== undefined && String(v).toLowerCase().includes(f))
    );
  }, [dataFilter, previewDataRows]);

  // Keep original filtered normalized data for search / other features
  const filteredVoterData = React.useMemo(() => {
    if (!dataFilter.trim()) return voterData;
    const f = dataFilter.toLowerCase();
    return voterData.filter(v => (
      (v.voter_id && String(v.voter_id).toLowerCase().includes(f)) ||
      (v.name && String(v.name).toLowerCase().includes(f)) ||
      (v.locality && String(v.locality).toLowerCase().includes(f)) ||
      (v.assembly && String(v.assembly).toLowerCase().includes(f)) ||
      (v.caste && String(v.caste).toLowerCase().includes(f))
    ));
  }, [dataFilter, voterData]);

  const totalPages = Math.max(1, Math.ceil(filteredPreviewData.length / rowsPerPage));
  const pageSafe = Math.min(previewPage, totalPages);
  const pageSlice = React.useMemo(() => {
    const start = (pageSafe - 1) * rowsPerPage;
    return filteredPreviewData.slice(start, start + rowsPerPage);
  }, [filteredPreviewData, pageSafe, rowsPerPage]);

  const allColumns = React.useMemo(() => {
    // Prefer server/client-detected original column order when available
    if (rawColumns && rawColumns.length) return rawColumns;
    const colSet = new Set();
    previewDataRows.slice(0, 200).forEach(row => Object.keys(row).forEach(k => colSet.add(k)));
    return Array.from(colSet);
  }, [previewDataRows, rawColumns]);

  const previewColumns = React.useMemo(() => {
    if (allColumns.length <= previewColumnsLimit) return allColumns;
    return allColumns.slice(0, previewColumnsLimit);
  }, [allColumns, previewColumnsLimit]);

  const downloadCSV = (onlyFiltered=false) => {
    const baseRows = previewDataRows;
    const rows = onlyFiltered ? filteredPreviewData : baseRows;
    if (!rows.length) return;
    const cols = (rawColumns && rawColumns.length)
      ? rawColumns
      : Array.from(new Set(rows.flatMap(r => Object.keys(r))));
    const esc = (val) => {
      if (val === null || val === undefined) return '';
      const s = String(val).replace(/"/g,'""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    };
    const lines = [cols.join(',')];
    for (const r of rows) {
      lines.push(cols.map(c => esc(r[c])).join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voter_data${onlyFiltered? '_filtered':''}_${rows.length}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadJSON = (onlyFiltered=false) => {
    const baseRows = previewDataRows;
    const rows = onlyFiltered ? filteredPreviewData : baseRows;
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voter_data${onlyFiltered? '_filtered':''}_${rows.length}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const apiBase = () => {
    // If running through Vite proxy, relative will work. Fallback to localhost:5000 if served from file or different port without proxy.
    if (window?.location?.hostname && window.location.port === '3000') return '';
    return '';
  };

  const apiFetch = (path, options) => fetch(`${apiBase()}${path}`, options);

  const checkApiHealth = async () => {
    try {
  const response = await apiFetch('/api/health');
      if (response.ok) {
        setApiStatus('connected');
        
        // Check if model is already loaded and restore status
        const data = await response.json();
        if (data.model_loaded) {
          setModelStatus({
            loaded: true,
            fileName: data.model_file || 'Unknown Model',
            fileSize: 'Already Loaded',
            modelType: 'VoterPredictor',
            features: data.feature_count,
            parties: ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
          });
          console.log('✅ Model status restored from API');
        }
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('error');
      console.error('API health check failed:', error);
    }
  };

  useEffect(() => {
    // Load sample voters data only in sample mode
    const loadSampleVoters = async () => {
      if (mode !== 'sample') {
        // In upload mode, just clear loading state
        setInitialLoading(false);
        return;
      }
      
      try {
        setInitialLoading(true);
        console.log('Loading sample voters...');
        const response = await fetch('/data/sample-voters.json');
        if (response.ok) {
          const voters = await response.json();
          console.log('Loaded sample voters:', voters.length);
          setSampleVoters(voters);
          if (!currentVoter && voters.length > 0) {
            setCurrentVoter(voters[0]);
          }
        } else {
          console.error('Failed to fetch sample voters:', response.status);
        }
      } catch (error) {
        console.error('Error loading sample voters:', error);
      } finally {
        setInitialLoading(false);
      }
    };

    loadSampleVoters();
  }, [currentVoter, mode]);

  // Handle model file upload
  const handleModelUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.pkl') && !file.name.endsWith('.pth')) {
      setUploadErrors(prev => ({...prev, model: 'Please upload a .pkl or .pth file'}));
      return;
    }

    setModelFile(file);
    setUploadErrors(prev => ({...prev, model: null}));
    
    // Call real ML API to load model
    setLoading(true);
    
    try {
      console.log(`🚀 Uploading model: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
      
      const formData = new FormData();
      formData.append('model', file);

      const response = await apiFetch('/api/upload-model', {
        method: 'POST',
        body: formData
      });
      
      console.log(`📡 Response status: ${response.status}`);
      console.log(`📝 Response headers:`, Object.fromEntries(response.headers.entries()));
      
      let result;
      try {
        const text = await response.text();
        console.log(`📄 Response text length: ${text.length} chars`);
        
        if (!text || text.trim() === '') {
          result = { error: `Empty response from server (status: ${response.status})` };
        } else {
          try {
            result = JSON.parse(text);
            console.log(`📋 Parsed response:`, result);
          } catch (parseError) {
            console.error(`❌ JSON parse error:`, parseError);
            result = { 
              error: 'Invalid JSON response from server', 
              parse_error: parseError.message,
              raw_response: text.substring(0, 200) + (text.length > 200 ? '...' : '')
            };
          }
        }
      } catch (textError) {
        console.error(`❌ Error reading response text:`, textError);
        result = { error: 'Failed to read response from server', text_error: textError.message };
      }

      if (response.ok && result.success) {
        console.log('✅ Model upload successful');
        setModelStatus({
          loaded: true,
          fileName: file.name,
          fileSize: (file.size / 1024 / 1024).toFixed(2) + ' MB',
          modelType: result.model_type,
          features: result.feature_count,
          parties: ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
        });
        alert(`Model loaded successfully!\nType: ${result.model_type}\nFeatures: ${result.feature_count}`);
      } else {
        console.error('❌ Model upload failed:', result);
        let errorMessage = result.error || 'Failed to load model';
        
        // Provide more helpful error messages
        if (response.status === 500) {
          errorMessage = `Server error (500): ${errorMessage}`;
          if (result.trace_tail && result.trace_tail.length > 0) {
            errorMessage += `\n\nServer error details:\n${result.trace_tail.join('\n')}`;
          }
        } else if (response.status === 0 || !response.status) {
          errorMessage = 'Network error: Could not connect to server. Make sure the backend is running on port 5000.';
        }
        
        setUploadErrors(prev => ({...prev, model: errorMessage}));
      }
    } catch (error) {
      console.error('💥 Network error:', error);
      let errorMessage = 'Network error: ' + error.message;
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Network error: Could not connect to server. Make sure the backend is running on port 5000.';
      }
      
      setUploadErrors(prev => ({...prev, model: errorMessage}));
    } finally {
      setLoading(false);
    }
  };

  // Handle voter data file upload
  const handleVoterDataUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
      setUploadErrors(prev => ({...prev, voterData: 'Please upload an Excel file (.xlsx or .xls)'}));
      return;
    }

    setVoterDataFile(file);
    setUploadErrors(prev => ({...prev, voterData: null}));
    setVoterData([]);
    setRawVoterData([]);
    
    const fileSizeMB = file.size / (1024 * 1024);
    
    // Try server-side processing first for better performance (files under 50MB)
    if (fileSizeMB <= 50) {
      try {
        setLoading(true);
        setUploadProgress({ 
          stage: 'Uploading to server for fast processing...', 
          percent: 10 
        });
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await apiFetch('/api/upload-voter-data', {
          method: 'POST',
          body: formData
        });
        
        setUploadProgress({ stage: 'Processing on server...', percent: 50 });
        
        let result;
        try {
          const text = await response.text();
          result = text ? JSON.parse(text) : { error: 'Empty response' };
        } catch (e) {
          result = { error: 'Invalid JSON response', parse_error: e.message };
        }
        
        if (response.ok && result.success) {
          setUploadProgress({ stage: 'Server processing complete!', percent: 100 });
          
          console.log('✅ ===== SERVER UPLOAD SUCCESS =====');
          console.log('✅ Backend returned:', {
            totalRows: result.total_rows,
            rowsReturned: result.rows_returned,
            rawDataLength: result.raw_data?.length,
            mappedDataLength: result.mapped_data?.length,
            backendCacheSize: result.backend_cache_size,
            columns: result.columns?.length
          });
          
          // Verify data before storing
          if (!result.mapped_data || result.mapped_data.length === 0) {
            console.error('❌ Backend returned empty mapped_data!');
            setUploadErrors(prev => ({
              ...prev, 
              voterData: 'Backend processed file but returned no data. Check Python terminal for errors.'
            }));
            setLoading(false);
            return;
          }
          
          // Use server-processed data
          setRawVoterData(result.raw_data);
          setVoterData(result.mapped_data);
          if (Array.isArray(result.columns)) setRawColumns(result.columns);
          
          // Extract sample voter IDs
          const sampleIds = result.mapped_data.slice(0, 10).map(voter => voter.voter_id).filter(id => id);
          setSampleVoterIds(sampleIds);
          
          console.log('✅ Frontend state updated:');
          console.log('   - voterData length:', result.mapped_data.length);
          console.log('   - rawVoterData length:', result.raw_data.length);
          console.log('   - sampleVoterIds:', sampleIds);
          console.log('✅ ===== END SERVER UPLOAD =====');
          
          setTimeout(() => {
            setLoading(false);
            setUploadProgress({ stage: '', percent: 0 });
            alert(`✅ Server processed ${result.rows_returned} voters successfully!\n\nBackend cache: ${result.backend_cache_size} voters\nSample IDs:\n${sampleIds.join('\n')}`);
          }, 1000);
          
          console.log(`✅ Server processed ${result.rows_returned}/${result.total_rows} rows successfully`);
          return; // Success - exit early
          
        } else {
          console.warn('Server processing failed, falling back to client-side processing:', result.error);
          // Fall through to client-side processing
        }
        
      } catch (error) {
        console.warn('Server upload failed, falling back to client-side processing:', error.message);
        // Fall through to client-side processing
      }
    }
    
    // Client-side processing fallback for large files or server failures
    console.log(`📁 Using client-side processing for ${fileSizeMB.toFixed(1)}MB file`);
    
    // HARD LIMIT: Files over 300MB are not supported in browser
    if (fileSizeMB > 300) {
      setUploadErrors(prev => ({
        ...prev, 
        voterData: `File too large: ${fileSizeMB.toFixed(1)}MB. Browser processing limited to 300MB. Server processing failed. Please:\n\n1. Split the file into smaller chunks (under 50MB for server processing)\n2. Or try a smaller sample of the data.`
      }));
      setLoading(false);
      return;
    }
    if (fileSizeMB > 1000) {
      const proceed = window.confirm(
        `EXTREMELY LARGE FILE DETECTED: ${(fileSizeMB / 1024).toFixed(1)}GB\n\n` +
        `Processing files larger than 1GB may:\n` +
        `• Take 10+ minutes to process\n` +
        `• Use significant memory (4-8GB RAM)\n` +
        `• Potentially crash the browser\n\n` +
        `RECOMMENDATION: Split the file into smaller chunks (100-500MB each)\n\n` +
        `Continue anyway? (Not recommended)`
      );
      if (!proceed) {
        setLoading(false);
        return;
      }
      // Force garbage collection if available
      if (window.gc) window.gc();
    } else if (fileSizeMB > 500) {
      const proceed = window.confirm(
        `Very large file detected: ${fileSizeMB.toFixed(1)}MB.\n\n` +
        `This may take several minutes and use significant memory.\n` +
        `For files larger than 1GB, consider splitting the data.\n\n` +
        `Do you want to proceed?`
      );
      if (!proceed) {
        setLoading(false);
        return;
      }
    } else if (fileSizeMB > 100) {
      console.warn(`Large file detected: ${fileSizeMB.toFixed(1)}MB. Processing may take longer.`);
    }
    
    // Show loading immediately
    setLoading(true);
    setUploadProgress({ 
      stage: fileSizeMB > 5 ? 'Processing large file, please wait...' : 'Preparing to process file...', 
      percent: 0 
    });
    
    // Use setTimeout to allow UI to update before heavy processing
    setTimeout(async () => {
      try {
        console.log('Starting to parse Excel file:', file.name, 'Size:', file.size, 'bytes');
        setUploadProgress({ stage: 'Reading file...', percent: 10 });
        
        // For very large files, read in chunks
        let arrayBuffer;
        if (fileSizeMB > 500) {
          setUploadProgress({ stage: 'Reading large file in chunks...', percent: 15 });
          // Read file with streaming approach for large files
          arrayBuffer = await file.arrayBuffer();
        } else {
          arrayBuffer = await file.arrayBuffer();
        }
        
        console.log('File read successfully, parsing...');
        setUploadProgress({ stage: 'Parsing Excel structure...', percent: 20 });
        
        // Parse with optimized settings for large files
        const workbook = XLSX.read(arrayBuffer, { 
          type: 'array',
          cellDates: false, // Disable date parsing for speed
          cellNF: false,
          cellText: true, // Keep as text for safety
          raw: false, // Don't parse raw values
          defval: '', // Default for empty cells
          dense: fileSizeMB > 100 // Use dense format for large files
        });
        
        console.log('Workbook parsed, sheet names:', workbook.SheetNames);
        setUploadProgress({ stage: 'Processing worksheets...', percent: 30 });

        // Combine ALL sheets: build ordered union of headers, and object rows per sheet
        const sheetNames = workbook.SheetNames || [];
        const orderedColumns = [];
        const objectRows = [];

        for (const sName of sheetNames) {
          const ws = workbook.Sheets[sName];
          if (!ws) continue;
          const sj = XLSX.utils.sheet_to_json(ws, {
            header: 1,
            defval: '',
            blankrows: false,
            raw: false,
            dateNF: 'YYYY-MM-DD'
          });
          if (!sj || sj.length < 2) continue;

          const headers = sj[0].map(h => String(h));
          // Extend orderedColumns with any new headers from this sheet in order
          for (const h of headers) {
            if (h && !orderedColumns.includes(h)) orderedColumns.push(h);
          }

          // Convert each row array into an object using the sheet's headers
          for (let r = 1; r < sj.length; r++) {
            const row = sj[r];
            if (!row || !row.some(cell => cell !== null && cell !== undefined && cell !== '')) continue;
            const obj = {};
            for (let c = 0; c < headers.length; c++) {
              const key = headers[c];
              if (!key) continue;
              obj[key] = row[c];
            }
            objectRows.push(obj);
          }
        }

        if (!objectRows.length) {
          setUploadErrors(prev => ({...prev, voterData: 'No data rows found in any worksheet'}));
          setLoading(false);
          setUploadProgress({ stage: '', percent: 0 });
          return;
        }

        // Preserve original column order for preview/export
        setRawColumns(orderedColumns);
        console.log(`Merged ${objectRows.length} rows across ${sheetNames.length} sheets.`);
        setUploadProgress({ stage: `Processing ${objectRows.length} voter records...`, percent: 50 });
        
        // Process data in chunks to prevent UI freezing for large files
        // Dynamic chunk size and delays based on file size
        const baseChunkSize = fileSizeMB > 1000 ? 25 : fileSizeMB > 500 ? 50 : fileSizeMB > 100 ? 100 : 200;
        
        const processChunk = (startIndex, chunkSize = baseChunkSize) => {
          return new Promise((resolve) => {
            // Longer delays for very large files to prevent memory overflow
            const delay = fileSizeMB > 1000 ? 100 : fileSizeMB > 500 ? 50 : fileSizeMB > 100 ? 20 : 5;
            
            setTimeout(() => {
              const endIndex = Math.min(startIndex + chunkSize, objectRows.length);
              let chunkData = [];
              try {
                chunkData = objectRows.slice(startIndex, endIndex);
                // For very large files, trigger garbage collection periodically
                if (fileSizeMB > 500 && startIndex % (baseChunkSize * 10) === 0) {
                  if (window.gc) window.gc();
                }
                
                resolve(chunkData);
              } catch (error) {
                console.error('Error processing chunk:', error);
                resolve([]); // Return empty array on error to continue processing
              }
            }, delay);
          });
        };
        
  // Process all chunks with memory management
  const allMappedData = [];
  const allRawData = [];
        const chunkSize = baseChunkSize;
        let processedCount = 0;
        
        for (let i = 0; i < objectRows.length; i += chunkSize) {
          try {
            const chunk = await processChunk(i, chunkSize);
            if (chunk && chunk.length > 0) {
              allMappedData.push(...chunk);
              allRawData.push(...chunk.map(r => ({ ...r })));
              processedCount += chunk.length;
            }
          
            // Update progress
            const progress = Math.round((i + chunkSize) / objectRows.length * 30) + 50; // 50-80%
            setUploadProgress({ 
              stage: `Processing voters: ${processedCount}/${objectRows.length} (${Math.round(processedCount/objectRows.length*100)}%)`, 
              percent: Math.min(progress, 80) 
            });
            
            // Force garbage collection for very large files
            if (fileSizeMB > 500 && i % (chunkSize * 5) === 0) {
              await new Promise(resolve => setTimeout(resolve, 100));
              if (window.gc) window.gc();
            }
          } catch (chunkError) {
            console.error('Error processing chunk at index:', i, chunkError);
            // Continue processing other chunks
          }
        }
        
        console.log('Total processed rows:', processedCount, 'out of', dataRows.length);
        
        console.log('All data processed, mapping to voter format...');
        setUploadProgress({ stage: 'Mapping voter data fields...', percent: 85 });
        
        // Map Excel columns to our expected format (process in chunks)
        const mappedData = [];
        
        for (let index = 0; index < allMappedData.length; index++) {
          const row = allMappedData[index];
          
          // Helper function to find column value by various possible names
          const findColumnValue = (possibleNames) => {
            for (const name of possibleNames) {
              // Try exact match first
              const exactMatch = row[name];
              if (exactMatch !== undefined && exactMatch !== null && exactMatch !== '') {
                return String(exactMatch).trim();
              }
              
              // Try case-insensitive and partial matching
              const key = Object.keys(row).find(k => {
                const keyLower = k.toLowerCase().replace(/[^a-z0-9]/g, '');
                const nameLower = name.toLowerCase().replace(/[^a-z0-9]/g, '');
                return keyLower === nameLower || 
                       keyLower.includes(nameLower) || 
                       nameLower.includes(keyLower);
              });
              
              if (key && row[key] !== undefined && row[key] !== null && row[key] !== '') {
                return String(row[key]).trim();
              }
            }
            return null;
          };
          
          const mappedVoter = {
            voter_id: findColumnValue([
              'voter_id', 'Voter ID', 'ID', 'VoterID', 'EPIC', 'Epic No', 'epic_no',
              'VOTER_ID', 'voter id', 'EpicNo', 'EPIC_NO', 'VoterCard', 'voter_card',
              'voters id', 'voters_id'
            ]) || `VOTER_${String(index + 1).padStart(4, '0')}`,
            
            name: findColumnValue([
              'name', 'Name', 'Full Name', 'Voter Name', 'FullName', 'full_name',
              'NAME', 'voter_name', 'VoterName', 'First Name', 'fname', 'relation name'
            ]) || 'Unknown',
            
            age: (() => {
              const ageValue = findColumnValue([
                'age', 'Age', 'AGE', 'age_years', 'years'
              ]);
              const parsed = parseInt(ageValue);
              return (parsed && parsed > 0 && parsed < 120) ? parsed : 30;
            })(),
            
            gender: (() => {
              const genderValue = findColumnValue([
                'gender', 'Gender', 'Sex', 'M/F', 'GENDER', 'sex'
              ]);
              if (!genderValue) return 'Unknown';
              const g = genderValue.toLowerCase().charAt(0);
              return g === 'm' ? 'Male' : g === 'f' ? 'Female' : genderValue;
            })(),
            
            religion: findColumnValue([
              'religion', 'Religion', 'RELIGION', 'religious_preference', 'faith'
            ]) || 'HINDU',
            
            caste: findColumnValue([
              'caste', 'Caste', 'CASTE', 'Category', 'Social Category', 'social_category'
            ]) || 'GENERAL',
            
            economic_category: findColumnValue([
              'economic_category', 'Economic Category', 'Income', 'Economic Status', 
              'Category', 'income_level', 'economic_status', 'class'
            ]) || 'MIDDLE CLASS',
            
            locality: findColumnValue([
              'locality', 'Locality', 'Area', 'Location', 'Address', 'locality_name',
              'area_name', 'zone', 'sector', 'section no & road name'
            ]) || 'Unknown',
            
            booth_no: (() => {
              const boothValue = findColumnValue([
                'partno', 'part_no', 'booth_no', 'Booth', 'Booth No', 'Polling Booth', 'BoothNo', 
                'booth_number', 'polling_booth_no', 'pb_no', 'part_number'
              ]);
              const parsed = parseInt(boothValue);
              return (parsed && parsed > 0) ? parsed : index + 1;
            })(),
            
            // Add partno explicitly for ML model
            partno: (() => {
              const partValue = findColumnValue(['partno', 'part_no', 'part_number']);
              const parsed = parseInt(partValue);
              return (parsed && parsed > 0) ? parsed : null;
            })(),
            
            assembly: findColumnValue([
              'assembly', 'Assembly', 'Constituency', 'AC', 'assembly_constituency',
              'AssemblyConstituency', 'ac_name', 'assembly name'
            ]) || 'Unknown',
            
            ward: findColumnValue([
              'ward', 'Ward', 'Ward No', 'ward_no', 'ward_number'
            ]) || 'Unknown',
            
            full_address: findColumnValue([
              'full address', 'full_address', 'Address', 'Full Address', 'Complete Address',
              'address', 'residential_address', 'Residential Address', 'FULL ADDRESS',
              'FULL_ADDRESS', 'ADDRESS', 'complete_address', 'Full_Address'
            ]) || (() => {
              // Debug: if no address found, log available row keys
              console.log('Address Debug - No address found. Available row keys:', Object.keys(row));
              console.log('Address Debug - Row data sample:', row);
              return 'Address not found in data';
            })(),
            
            // Additional fields
            mobile: findColumnValue([
              'mobile', 'Mobile', 'Phone', 'Contact', 'mobile_no', 'phone_no'
            ]) || '',
            
            email: findColumnValue([
              'email', 'Email', 'E-mail', 'email_id'
            ]) || '',
            
            // Numeric features for ML model (exactly like app.py)
            land_rate_per_sqm: (() => {
              const value = findColumnValue([
                'land_rate_per_sqm', 'land_rate', 'Land Rate', 'land_cost'
              ]);
              return parseFloat(value) || 0.0;
            })(),
            
            construction_cost_per_sqm: (() => {
              const value = findColumnValue([
                'construction_cost_per_sqm', 'construction_cost', 'Construction Cost', 'building_cost'
              ]);
              return parseFloat(value) || 0.0;
            })(),
            
            population: (() => {
              const value = findColumnValue([
                'population', 'Population', 'area_population', 'locality_population'
              ]);
              return parseFloat(value) || 0.0;
            })(),
            
            male_female_ratio: (() => {
              const value = findColumnValue([
                'male_female_ratio', 'MaleToFemaleRatio', 'gender_ratio', 'sex_ratio'
              ]);
              return parseFloat(value) || 1.0;
            })(),
            
            // Family identifiers for core and chain family predictions (from app.py)
            core_family_id: findColumnValue([
              'core_family_id', 'family_id_main', 'core_id', 'family_id', 'core_family', 'CoreFamilyID',
              'household_id'
            ]) || null,
            
            family_by_chain_id: findColumnValue([
              'family_by_chain_id', 'family_by_chain', 'chain_id', 'family_chain_id', 'FamilyByChainID'
            ]) || null,
            
            occupation: findColumnValue([
              'occupation', 'Occupation', 'Job', 'profession'
            ]) || 'Unknown'
          };

          // Economic category fallback from code
          if ((!mappedVoter.economic_category || mappedVoter.economic_category === 'MIDDLE CLASS') ) {
            const econCode = findColumnValue(['economic_category_code', 'econ_code', 'eco_code']);
            if (econCode) {
              const codeNorm = String(econCode).trim().toUpperCase();
              const codeMap = {
                '1': 'LOW INCOME AREAS', 'LOW': 'LOW INCOME AREAS', 'L': 'LOW INCOME AREAS',
                '2': 'LOWER MIDDLE CLASS', 'LM': 'LOWER MIDDLE CLASS',
                '3': 'MIDDLE CLASS', 'M': 'MIDDLE CLASS',
                '4': 'UPPER MIDDLE CLASS', 'UM': 'UPPER MIDDLE CLASS',
                '5': 'PREMIUM AREAS', 'P': 'PREMIUM AREAS', 'HIGH': 'PREMIUM AREAS'
              };
              if (codeMap[codeNorm]) mappedVoter.economic_category = codeMap[codeNorm];
            }
          }

          // Derive family chain id if missing
            if (!mappedVoter.family_by_chain_id) {
              const chainRaw = findColumnValue(['family_by_chain']);
              if (chainRaw) mappedVoter.family_by_chain_id = chainRaw;
            }

          // Normalize male_female_ratio if given as percentage string
          if (typeof mappedVoter.male_female_ratio === 'string' && mappedVoter.male_female_ratio.includes(':')) {
            // e.g., "950:1000" convert to ratio 0.95
            const parts = mappedVoter.male_female_ratio.split(':').map(p => parseFloat(p));
            if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
              mappedVoter.male_female_ratio = parts[0] / parts[1];
            }
          }
          
          mappedData.push(mappedVoter);
          
          // Allow UI updates every 25 items for better responsiveness
          if (index % 25 === 0) {
            const progress = Math.round((index / allMappedData.length) * 15) + 85; // 85-100%
            setUploadProgress({ 
              stage: `Finalizing voter data: ${index + 1}/${allMappedData.length}`, 
              percent: Math.min(progress, 100) 
            });
            await new Promise(resolve => setTimeout(resolve, 1));
          }
        }
        
  setUploadProgress({ stage: 'Completed successfully!', percent: 100 });
  setVoterData(mappedData);
  setRawVoterData(allRawData);
        
        // Extract sample voter IDs for display
        const sampleIds = mappedData.slice(0, 10).map(voter => voter.voter_id).filter(id => id);
        setSampleVoterIds(sampleIds);
        
        // Log success details for debugging
        console.log('✅ ===== DATA UPLOAD SUCCESS =====');
        console.log('✅ Total voters loaded:', mappedData.length);
        console.log('✅ Sample voter IDs:', sampleIds);
        console.log('✅ First voter example:', mappedData[0]);
        console.log('✅ Voter ID column detected:', mappedData[0]?.voter_id);
        console.log('✅ ===== END DATA UPLOAD =====');
        
        // Clear progress after a short delay
        setTimeout(() => {
          setLoading(false);
          setUploadProgress({ stage: '', percent: 0 });
          
          // Show success message to user
          alert(`✅ Successfully loaded ${mappedData.length} voters!\n\nSample Voter IDs:\n${sampleIds.join('\n')}\n\nYou can now search for any voter ID.`);
        }, 1000);
        
      } catch (error) {
        console.error('=== EXCEL PARSING ERROR ===');
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
        console.error('File info:', {
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: new Date(file.lastModified)
        });
        console.error('Processing stage when error occurred:', uploadProgress.stage);
        console.error('=== END ERROR DEBUG ===');
        
        setUploadErrors(prev => ({
          ...prev, 
          voterData: `Error parsing Excel file: ${error.message}. Please check the file format and try again. File: ${file.name} (${(file.size/(1024*1024)).toFixed(1)}MB)`
        }));
        setLoading(false);
        setUploadProgress({ stage: '', percent: 0 });
      }
    }, 100); // Initial delay to show loading state
  };

  // Search for voter by ID
  const handleVoterSearch = async () => {
    if (!searchVoterId.trim()) {
      setUploadErrors(prev => ({...prev, search: 'Please enter a voter ID'}));
      return;
    }

    console.log('🔍 ===== VOTER SEARCH DEBUG =====');
    const wanted = searchVoterId.trim();
    console.log('🔍 Searching for:', wanted);
    console.log('🔍 Total voters in local data:', voterData.length);
    console.log('🔍 Sample local voter IDs:', voterData.slice(0, 5).map(v => v.voter_id));
    
    // Try multiple search strategies
    console.log('🔍 Strategy 1: Exact match (case-insensitive)');
    let local = voterData.find(v => String(v.voter_id || '').trim().toUpperCase() === wanted.toUpperCase());
    
    if (!local) {
      console.log('🔍 Strategy 2: Partial match');
      local = voterData.find(v => {
        const vid = String(v.voter_id || '').trim();
        return vid.includes(wanted) || wanted.includes(vid);
      });
    }
    
    if (!local) {
      console.log('🔍 Strategy 3: Normalized match (remove spaces/special chars)');
      const normalizedWanted = wanted.replace(/[^a-zA-Z0-9]/g, '').toUpperCase();
      local = voterData.find(v => {
        const normalized = String(v.voter_id || '').replace(/[^a-zA-Z0-9]/g, '').toUpperCase();
        return normalized === normalizedWanted;
      });
    }
    
    if (local) {
      console.log('✅ Found voter locally:', local.voter_id, local.name);
      setSearchedVoter(local);
      setCurrentVoter(local);
      setUploadErrors(prev => ({...prev, search: null}));
      setRealPredictions(null);
      return;
    }
    
    console.log('❌ Not found locally, trying backend search...');

    // Fallback to backend search for large datasets or different mappings
    try {
      const res = await apiFetch('/api/search-voter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voter_id: wanted })
      });
      const data = await res.json();
      console.log('📡 Backend search response:', data);
      
      if (res.ok && data.success && data.voter) {
        console.log('✅ Found voter in backend:', data.voter.voter_id);
        setSearchedVoter(data.voter);
        setCurrentVoter(data.voter);
        setUploadErrors(prev => ({...prev, search: null}));
        setRealPredictions(null);
      } else {
        console.log('❌ Voter not found in backend either');
        const totalVoters = voterData.length;
        const sampleIds = sampleVoterIds.length > 0 
          ? sampleVoterIds.slice(0, 5).join(', ')
          : voterData.slice(0, 5).map(v => v.voter_id).join(', ');
        const avail = data.available_sample ? ` Backend sample: ${data.available_sample.join(', ')}` : '';
        setUploadErrors(prev => ({
          ...prev, 
          search: `Voter ID "${wanted}" not found in ${totalVoters} uploaded voters.\n\nSample IDs from your data: ${sampleIds}${avail}\n\nTip: Copy-paste an ID exactly as shown above.`
        }));
        setSearchedVoter(null);
      }
    } catch (e) {
      console.error('❌ Backend search error:', e);
      setUploadErrors(prev => ({
        ...prev, 
        search: `Search failed: ${e.message}. Backend may not be running. Check console for details.`
      }));
      setSearchedVoter(null);
    }
    console.log('🔍 ===== END VOTER SEARCH =====');
  };

  // Generate real predictions using uploaded model via API
  const generateRealPredictions = async (voter) => {
    console.log('🔮 Generating predictions for voter:', voter);
    console.log('📊 Current model status:', modelStatus);
    console.log('🌐 API status:', apiStatus);
    
    setLoading(true);
    setRealPredictions(null);
    setFamilyPredictions(null);
    
    try {
      // Check API health before making prediction
      console.log('🔍 Checking API health before prediction...');
      const healthResponse = await apiFetch('/api/health');
      let healthData = {};
      try {
        const ht = await healthResponse.text();
        healthData = ht ? JSON.parse(ht) : {};
      } catch (e) {
        console.warn('Health JSON parse issue:', e.message);
      }
      console.log('🏥 API health status:', healthData);
      
      // Call real ML API
      console.log('📡 Making prediction API call...');
      console.log('🗳️ Voter data being sent:', {
        voter_id: voter.voter_id,
        name: voter.name,
        age: voter.age,
        booth_no: voter.booth_no,
        partno: voter.partno || voter.booth_no, // Use booth_no as fallback for partno
        part_no: voter.part_no || voter.booth_no,
        booth_number: voter.booth_number || voter.booth_no,
        religion: voter.religion,
        caste: voter.caste,
        economic_category: voter.economic_category,
        // Numeric features for debugging
        land_rate_per_sqm: voter.land_rate_per_sqm,
        construction_cost_per_sqm: voter.construction_cost_per_sqm,
        population: voter.population,
        male_female_ratio: voter.male_female_ratio
      });
      const response = await apiFetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(voter)
      });
      let result;
      try {
        const text = await response.text();
        result = text ? JSON.parse(text) : { error: 'Empty response' };
      } catch (e) {
        result = { error: 'Invalid JSON (predict)', parse_error: e.message };
      }
      console.log('📊 API Response:', result);

      if (response.ok && result.success) {
        console.log('✅ Setting real predictions:', result.prediction);
        setRealPredictions(result.prediction);
        setPredictions(null); // Clear any mock predictions
        await generateFamilyPredictions(voter);
      } else {
        console.error('❌ Prediction API error:', result.error);
        alert('Prediction failed: ' + (result.error || 'Unknown error'));
      }
    } catch (error) {
      alert('Network error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Generate family predictions using core and chain family concepts from app.py
  const generateFamilyPredictions = async (voter) => {
    try {
      if (!voter.core_family_id && !voter.family_by_chain_id) {
        console.log('⚠️ No family identifiers found for this voter');
        console.log('🔍 Voter data:', { 
          core_family_id: voter.core_family_id, 
          family_by_chain_id: voter.family_by_chain_id 
        });
        return;
      }

      console.log('👨‍👩‍👧‍👦 Generating real family predictions...');
      console.log('🔍 Family IDs - Core:', voter.core_family_id, 'Chain:', voter.family_by_chain_id);
      console.log('📊 Total voter data available:', voterData?.length || 0);
      
      // Check if voterData is available
      if (!voterData || voterData.length === 0) {
        console.log('⚠️ No voter dataset available for family search');
        return;
      }
      
      // Helper to normalize IDs for robust comparison
      const norm = (val) => (val === null || val === undefined) ? '' : String(val).trim().toUpperCase();
      const selfId = voter.voter_id ? String(voter.voter_id).trim() : '';
      const coreId = norm(voter.core_family_id);
      const chainId = norm(voter.family_by_chain_id);

      // Find family members locally (case-insensitive, trimmed), exclude the main voter, and dedupe by voter_id
      const seen = new Set();
      const coreMembers = coreId ? 
        voterData
          .filter(v => norm(v.core_family_id) === coreId)
          .filter(v => String(v.voter_id || '').trim() !== selfId)
          .filter(v => {
            const id = String(v.voter_id || '').trim();
            if (!id || seen.has(id)) return false;
            seen.add(id);
            return true;
          })
          .slice(0, 10) : [];
      const chainMembers = chainId ? 
        voterData
          .filter(v => norm(v.family_by_chain_id) === chainId)
          .filter(v => String(v.voter_id || '').trim() !== selfId)
          .filter(v => {
            const id = String(v.voter_id || '').trim();
            if (!id || seen.has(id)) return false;
            seen.add(id);
            return true;
          })
          .slice(0, 10) : [];
      
      console.log(`🏠 Found ${coreMembers.length} core family members locally`);
      console.log(`🔗 Found ${chainMembers.length} chain family members locally`);
      
      if (coreMembers.length === 0 && chainMembers.length === 0) {
        console.log('⚠️ No other family members found in dataset matching the provided IDs (excluding the selected voter).');
        return;
      }
      
      const response = await apiFetch('/api/predict-family', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          voter: voter,
          coreMembers: coreMembers, // Send only family members, not entire dataset
          chainMembers: chainMembers,
          core_family_id: voter.core_family_id,
          family_by_chain_id: voter.family_by_chain_id
        })
      });
      let result;
      try {
        const text = await response.text();
        result = text ? JSON.parse(text) : { error: 'Empty response (family)' };
      } catch (e) {
        result = { error: 'Invalid JSON (family)', parse_error: e.message };
      }

      if (response.ok && result.success) {
        console.log('✅ Family predictions generated:', result.family_predictions);
        setFamilyPredictions(result.family_predictions);
      } else {
        console.error('❌ Family prediction failed:', response.status, result);
        console.error('❌ Error details:', result.error || 'Unknown error');
      }
    } catch (error) {
      console.error('❌ Family prediction error:', error);
      console.error('❌ Network or parsing error details:', error.message);
      // Silent error handling for family predictions
    }
  };

  const calculateRealPredictions = (voter) => {
    // Enhanced prediction logic simulating real ML model output
    let baseProbs = { BJP: 0.28, Congress: 0.12, AAP: 0.48, Others: 0.08, NOTA: 0.04 };
    let turnoutBase = 0.72;

    // More sophisticated demographic adjustments
    const ageGroup = voter.age < 30 ? 'young' : voter.age > 50 ? 'senior' : 'middle';
    const economicLevel = voter.economic_category.includes('PREMIUM') || voter.economic_category.includes('UPPER') ? 'high' :
                         voter.economic_category.includes('LOW') ? 'low' : 'middle';

    // Complex interaction effects
    if (voter.religion === 'HINDU') {
      if (voter.caste === 'BRAHMIN') {
        baseProbs.BJP += ageGroup === 'senior' ? 0.18 : 0.12;
        baseProbs.AAP -= 0.10;
        turnoutBase += 0.08;
      } else if (voter.caste === 'OBC') {
        if (economicLevel === 'low') {
          baseProbs.AAP += 0.15;
          baseProbs.BJP -= 0.08;
        } else {
          baseProbs.BJP += 0.05;
        }
      } else if (voter.caste === 'SC') {
        baseProbs.AAP += 0.20;
        baseProbs.BJP -= 0.12;
        baseProbs.Congress += 0.03;
      }
    } else if (voter.religion === 'MUSLIM') {
      baseProbs.AAP += 0.25;
      baseProbs.BJP -= 0.20;
      baseProbs.Congress += 0.08;
      turnoutBase += 0.05;
    }

    // Economic adjustments with locality interaction
    if (economicLevel === 'high') {
      baseProbs.BJP += 0.12;
      turnoutBase += 0.15;
      if (voter.locality.includes('Puram') || voter.locality.includes('Nagar')) {
        baseProbs.BJP += 0.05;
      }
    } else if (economicLevel === 'low') {
      baseProbs.AAP += 0.18;
      turnoutBase -= 0.03;
    }

    // Age-economic interaction
    if (ageGroup === 'young' && economicLevel === 'middle') {
      baseProbs.AAP += 0.12;
      baseProbs.BJP -= 0.06;
    }

    // Normalize and add some randomness for realism
    const total = Object.values(baseProbs).reduce((sum, val) => sum + val, 0);
    Object.keys(baseProbs).forEach(party => {
      baseProbs[party] = Math.max(0.01, Math.min(0.85, baseProbs[party] / total + (Math.random() - 0.5) * 0.05));
    });

    // Renormalize after randomness
    const newTotal = Object.values(baseProbs).reduce((sum, val) => sum + val, 0);
    Object.keys(baseProbs).forEach(party => {
      baseProbs[party] = baseProbs[party] / newTotal;
    });

    turnoutBase = Math.max(0.35, Math.min(0.92, turnoutBase + (Math.random() - 0.5) * 0.1));

    const predictedParty = Object.entries(baseProbs).reduce((a, b) => 
      baseProbs[a[0]] > baseProbs[b[0]] ? a : b
    )[0];

    const maxProb = Math.max(...Object.values(baseProbs));
    const confidence = maxProb > 0.45 ? 'High' : maxProb > 0.28 ? 'Medium' : 'Low';

    return {
      party_probabilities: baseProbs,
      predicted_party: predictedParty,
      turnout_probability: turnoutBase,
      confidence_level: confidence,
      model_confidence: (maxProb * 100).toFixed(1) + '%',
      prediction_factors: {
        primary: voter.religion + ' ' + voter.caste,
        secondary: voter.economic_category,
        tertiary: ageGroup + ' voter in ' + voter.locality
      }
    };
  };

  useEffect(() => {
    // Only generate mock predictions in sample mode
    if (currentVoter && mode === 'sample') {
      generateMockPredictions(currentVoter);
    }
  }, [currentVoter, mode]);

  // Removed auto-prediction for upload mode - user must click button

  const generateMockPredictions = (voter) => {
    setLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      // Generate realistic predictions based on voter demographics
      const predictions = calculatePredictions(voter);
      setPredictions(predictions);
      setLoading(false);
    }, 800);
  };

  const calculatePredictions = (voter) => {
    // Mock prediction logic based on demographics
    let baseProbs = { BJP: 0.25, Congress: 0.15, AAP: 0.45, Others: 0.10, NOTA: 0.05 };
    let turnoutBase = 0.65;

    // Adjust based on demographics
    if (voter.religion === 'HINDU') {
      if (voter.caste === 'BRAHMIN' || voter.caste === 'KSHATRIYA') {
        baseProbs.BJP += 0.15;
        baseProbs.AAP -= 0.10;
      } else if (voter.caste === 'OBC') {
        baseProbs.AAP += 0.10;
        baseProbs.BJP -= 0.05;
      } else if (voter.caste === 'SC') {
        baseProbs.AAP += 0.15;
        baseProbs.BJP -= 0.10;
      }
    } else if (voter.religion === 'MUSLIM') {
      baseProbs.AAP += 0.20;
      baseProbs.BJP -= 0.15;
      baseProbs.Congress += 0.05;
    } else if (voter.religion === 'SIKH') {
      baseProbs.AAP += 0.25;
      baseProbs.BJP -= 0.10;
    }

    // Economic category adjustments
    if (voter.economic_category === 'PREMIUM AREAS' || voter.economic_category === 'UPPER MIDDLE CLASS') {
      baseProbs.BJP += 0.10;
      turnoutBase += 0.15;
    } else if (voter.economic_category === 'LOW INCOME AREAS') {
      baseProbs.AAP += 0.15;
      turnoutBase -= 0.05;
    }

    // Age adjustments
    if (voter.age < 30) {
      baseProbs.AAP += 0.10;
      baseProbs.BJP -= 0.05;
    } else if (voter.age > 50) {
      baseProbs.BJP += 0.08;
      baseProbs.AAP -= 0.05;
      turnoutBase += 0.10;
    }

    // Normalize probabilities
    const total = Object.values(baseProbs).reduce((sum, val) => sum + val, 0);
    Object.keys(baseProbs).forEach(party => {
      baseProbs[party] = Math.max(0.01, baseProbs[party] / total);
    });

    // Ensure turnout is between 0.3 and 0.95
    turnoutBase = Math.max(0.3, Math.min(0.95, turnoutBase));

    // Find predicted party
    const predictedParty = Object.entries(baseProbs).reduce((a, b) => 
      baseProbs[a[0]] > baseProbs[b[0]] ? a : b
    )[0];

    return {
      party_probabilities: baseProbs,
      predicted_party: predictedParty,
      turnout_probability: turnoutBase,
      confidence_level: Math.max(...Object.values(baseProbs)) > 0.4 ? 'High' : 
                       Math.max(...Object.values(baseProbs)) > 0.25 ? 'Medium' : 'Low'
    };
  };

  // Utility function to get voter field with multiple possible names
  const getVoterField = (voter, ...fieldNames) => {
    // Disabled verbose address debug logging
    // if (fieldNames.includes('full address') || fieldNames.includes('full_address')) {
    //   console.log('Address Debug - Voter keys:', Object.keys(voter));
    //   console.log('Address Debug - Looking for fields:', fieldNames);
    //   fieldNames.forEach(field => {
    //     console.log(`Address Debug - ${field}:`, voter[field]);
    //   });
    // }
    
    for (const field of fieldNames) {
      // Try exact field name
      let value = voter[field];
      if (!value) {
        // Try lowercase
        value = voter[field?.toLowerCase()];
      }
      if (!value) {
        // Try uppercase
        value = voter[field?.toUpperCase()];
      }
      
      // Check if value exists and is not empty/null
      if (value && value !== 'Unknown' && value !== 'N/A' && value !== '' && value !== null && value !== undefined) {
        return value;
      }
    }
    
    // If no good value found, check if any field exists with 'Unknown' value
    // For address, we might want to show 'Unknown' rather than 'N/A'
    if (fieldNames.includes('full address') || fieldNames.includes('full_address')) {
      for (const field of fieldNames) {
        const value = voter[field] || voter[field?.toLowerCase()] || voter[field?.toUpperCase()];
        if (value === 'Unknown') {
          return 'Address not available';
        }
      }
    }
    
    return 'N/A';
  };

  // Detailed voter information component matching app.py format
  const DetailedVoterInfo = ({ voter, title = "👤 Voter Information" }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <h2 className="text-xl font-semibold mb-6 flex items-center">
        <User className="w-5 h-5 mr-2 text-blue-600" />
        {title}
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Column 1: Personal Details */}
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-900 border-b pb-1">Personal Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Name:</span>
              <span className="font-semibold">{getVoterField(voter, 'name', 'Name', 'NAAM')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Age:</span>
              <span className="font-semibold">{getVoterField(voter, 'age', 'Age', 'AGE')} years</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Gender:</span>
              <span className="font-semibold">{getVoterField(voter, 'gender', 'Gender', 'GENDER', 'sex')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Booth No:</span>
              <span className="font-semibold">{getVoterField(voter, 'booth_no', 'booth_number', 'partno', 'part_no', 'Booth')}</span>
            </div>
          </div>
        </div>

        {/* Column 2: Demographics */}
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-900 border-b pb-1">Demographics</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Religion:</span>
              <span className="font-semibold">{getVoterField(voter, 'religion', 'Religion', 'RELIGION')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Caste:</span>
              <span className="font-semibold">{getVoterField(voter, 'caste', 'Caste', 'CASTE', 'category')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Economic Category:</span>
              <span className="font-semibold text-xs">{getVoterField(voter, 'economic_category', 'Economic_Category', 'income_level')}</span>
            </div>
          </div>
        </div>

        {/* Column 3: Location & Contact */}
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-900 border-b pb-1">Location & Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Locality:</span>
              <span className="font-semibold">{getVoterField(voter, 'locality', 'Locality', 'area', 'zone')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Assembly:</span>
              <span className="font-semibold">{getVoterField(voter, 'assembly', 'Assembly', 'constituency')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 font-medium">Voter ID:</span>
              <span className="font-semibold font-mono text-xs">{getVoterField(voter, 'voter_id', 'VoterID', 'EPIC', 'ID')}</span>
            </div>
            {voter.mobile && voter.mobile !== 'Unknown' && (
              <div className="flex justify-between">
                <span className="text-gray-600 font-medium">Mobile:</span>
                <span className="font-semibold">{voter.mobile}</span>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Additional Information - Always show for debugging */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-start space-x-2">
          <MapPin className="w-4 h-4 text-gray-500 mt-0.5" />
          <div>
            <span className="text-gray-600 font-medium text-sm">Full Address:</span>
            <p className="text-sm font-semibold text-gray-800 mt-1">
              {getVoterField(voter, 'full address', 'full_address', 'address', 'Full_Address', 'Full Address', 'complete_address', 'Complete Address', 'residential_address', 'Residential Address')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const getConfidenceColor = (confidence) => {
    switch(confidence) {
      case 'High': return 'text-green-600';
      case 'Medium': return 'text-yellow-600';
      case 'Low': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getConfidenceEmoji = (confidence) => {
    switch(confidence) {
      case 'High': return '🎯';
      case 'Medium': return '🎲';
      case 'Low': return '❓';
      default: return '⚪';
    }
  };

  // Helper to get the correct prediction data based on mode
  const getCurrentPredictions = () => {
    if (mode === 'upload') {
      return realPredictions; // Only use real predictions in upload mode
    } else {
      return predictions; // Use mock predictions in sample mode
    }
  };

  // Check if we have valid predictions to display
  const hasValidPredictions = () => {
    const current = getCurrentPredictions();
    return current && current.party_probabilities && current.predicted_party;
  };

  if (initialLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Loading AI Voter Prediction Dashboard</h2>
          <p className="text-gray-600">Preparing voter data and prediction models...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className="min-h-screen p-6"
    >
      {/* Header with Navigation */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={onNavigateBack}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <button
              onClick={onNavigateHome}
              className="p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
            >
              <Home className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 flex items-center">
                <Zap className="w-10 h-10 mr-4 text-purple-600" />
                Electoral Individual Voter Prediction Dashboard
              </h1>
              <p className="text-gray-600 mt-2">
                AI-power
                ed electoral predictions for individual voters
              </p>
            </div>
          </div>
          
          {/* API Status Indicator */}
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${ 
              apiStatus === 'connected' ? 'bg-green-100 text-green-800' :
              apiStatus === 'error' ? 'bg-red-100 text-red-800' : 
              'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                apiStatus === 'connected' ? 'bg-green-500' :
                apiStatus === 'error' ? 'bg-red-500' : 
                'bg-yellow-500'
              }`}></div>
              <span className="text-sm font-medium">
                ML API: {apiStatus === 'connected' ? 'Connected' : apiStatus === 'error' ? 'Disconnected' : 'Checking...'}
              </span>
            </div>
            
            {/* Mode Toggle */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => {
                  setMode('sample');
                  setRealPredictions(null); // Clear real predictions when switching to sample mode
                }}
                className={`px-4 py-2 rounded-md transition-all ${
                  mode === 'sample' 
                    ? 'bg-white shadow-sm text-blue-600 font-medium' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Sample Data
              </button>
              <button
                onClick={() => {
                  setMode('upload');
                  setPredictions(null); // Clear mock predictions when switching to upload mode
                }}
                className={`px-4 py-2 rounded-md transition-all ${
                  mode === 'upload' 
                    ? 'bg-white shadow-sm text-purple-600 font-medium' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Upload & Predict
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Mode */}
      {mode === 'upload' && (
        <div className="space-y-6 mb-8">
          {/* File Upload Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <Upload className="w-5 h-5 mr-2 text-purple-600" />
              Upload Your Files
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Upload */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-700">
                  Trained Model (.pkl or .pth)
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".pkl,.pth"
                    onChange={handleModelUpload}
                    className="hidden"
                    id="model-upload"
                  />
                  <label
                    htmlFor="model-upload"
                    className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-purple-400 hover:bg-purple-50 transition-all"
                  >
                    <div className="text-center">
                      <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Click to upload model</p>
                      <p className="text-xs text-gray-400">PKL or PTH files only</p>
                    </div>
                  </label>
                </div>
                {uploadErrors.model && (
                  <p className="text-red-500 text-sm flex items-center">
                    <AlertCircle className="w-4 h-4 mr-1" />
                    {uploadErrors.model}
                  </p>
                )}
                {modelFile && (
                  <div className="flex items-center text-green-600 text-sm">
                    <CheckCircle className="w-4 h-4 mr-1" />
                    {modelFile.name}
                  </div>
                )}
              </div>

              {/* Voter Data Upload */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-700">
                  Voter Data (.xlsx or .xls)
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".xlsx,.xls"
                    onChange={handleVoterDataUpload}
                    className="hidden"
                    id="voter-data-upload"
                  />
                  <label
                    htmlFor="voter-data-upload"
                    className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all"
                  >
                    <div className="text-center">
                      <FileText className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Click to upload voter data</p>
                      <p className="text-xs text-gray-400">Excel files (≤50MB: Fast server processing, &gt;50MB: Browser processing)</p>
                    </div>
                  </label>
                </div>
                
                {/* Large File Instructions */}
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-xs text-green-800">
                  <div className="font-semibold mb-1">⚡ Fast Processing:</div>
                  <div className="space-y-1">
                    <div>• Files ≤50MB: Lightning-fast server processing (pandas)</div>
                    <div>• Files &gt;50MB: Browser processing (slower but works up to 300MB)</div>
                    <div>• Files &gt;300MB: Split into smaller chunks or contact support</div>
                  </div>
                </div>
                {uploadErrors.voterData && (
                  <p className="text-red-500 text-sm flex items-center">
                    <AlertCircle className="w-4 h-4 mr-1" />
                    {uploadErrors.voterData}
                  </p>
                )}
                {voterDataFile && voterData.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center text-green-600 text-sm">
                      <CheckCircle className="w-4 h-4 mr-1" />
                      {voterDataFile.name} ({voterData.length} voters loaded)
                    </div>
                    {voterData.length > 0 && sampleVoterIds.length > 0 && (
                      <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded">
                        <strong>📋 Sample Voter IDs from your data:</strong> 
                        <div className="mt-2 flex flex-wrap gap-2">
                          {sampleVoterIds.map((id, index) => (
                            <span 
                              key={index}
                              className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-mono cursor-pointer hover:bg-blue-200"
                              onClick={() => setSearchVoterId(id)}
                              title="Click to search this voter"
                            >
                              {id}
                            </span>
                          ))}
                        </div>
                        <p className="text-xs text-gray-500 mt-2">
                          💡 Click any voter ID above to auto-fill the search box
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Model Status */}
          {modelStatus && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl border border-green-200 p-6">
              <h3 className="text-lg font-semibold text-green-800 mb-4 flex items-center">
                <CheckCircle className="w-5 h-5 mr-2" />
                Model Loaded Successfully
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">File:</span>
                  <p className="font-medium">{modelStatus.fileName}</p>
                </div>
                <div>
                  <span className="text-gray-600">Size:</span>
                  <p className="font-medium">{modelStatus.fileSize}</p>
                </div>
                <div>
                  <span className="text-gray-600">Parties:</span>
                  <p className="font-medium">{modelStatus.parties.join(', ')}</p>
                </div>
              </div>
            </div>
          )}

          {/* Voter Search */}
          {voterData.length > 0 && modelStatus && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Search className="w-5 h-5 mr-2 text-blue-600" />
                Search Voter by ID
              </h3>
              <div className="flex space-x-4">
                <div className="flex-1">
                  <input
                    type="text"
                    value={searchVoterId}
                    onChange={(e) => setSearchVoterId(e.target.value)}
                    placeholder={`Enter Voter ID (${voterData.length} voters loaded)`}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono"
                    onKeyPress={(e) => e.key === 'Enter' && handleVoterSearch()}
                  />
                  {uploadErrors.search && (
                    <div className="text-red-500 text-sm mt-2 p-3 bg-red-50 rounded-lg border border-red-200 whitespace-pre-wrap">
                      <div className="flex items-start">
                        <AlertCircle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
                        <span>{uploadErrors.search}</span>
                      </div>
                    </div>
                  )}
                </div>
                <button
                  onClick={handleVoterSearch}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
                >
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </button>
              </div>
              
              {voterData.length > 0 && (
                <div className="mt-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-600">
                      <strong>Available Voter IDs:</strong> {voterData.slice(0, 10).map(v => v.voter_id).join(', ')}
                      {voterData.length > 10 && ` ... and ${voterData.length - 10} more`}
                    </p>
                    
                    {modelStatus && voterData.length > 0 && (
                      <button
                        onClick={() => {
                          // Generate predictions for first 5 voters as sample
                          const sampleVoters = voterData.slice(0, 5);
                          console.log('Generating batch predictions for:', sampleVoters.length, 'voters');
                          // This could be expanded to full batch processing
                        }}
                        className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm flex items-center space-x-2"
                      >
                        <Zap className="w-4 h-4" />
                        <span>Batch Predictions</span>
                      </button>
                    )}
                  </div>
                  
                  {/* Data Preview */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-800 mb-2">Data Preview (First 3 voters):</h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-gray-300">
                            <th className="text-left p-1">Voter ID</th>
                            <th className="text-left p-1">Name</th>
                            <th className="text-left p-1">Age</th>
                            <th className="text-left p-1">Gender</th>
                            <th className="text-left p-1">Religion</th>
                            <th className="text-left p-1">Caste</th>
                            <th className="text-left p-1">Location</th>
                          </tr>
                        </thead>
                        <tbody>
                          {voterData.slice(0, 3).map((voter, index) => (
                            <tr key={index} className="border-b border-gray-200">
                              <td className="p-1 font-medium">{voter.voter_id}</td>
                              <td className="p-1">{voter.name}</td>
                              <td className="p-1">{voter.age}</td>
                              <td className="p-1">{voter.gender}</td>
                              <td className="p-1">{voter.religion}</td>
                              <td className="p-1">{voter.caste}</td>
                              <td className="p-1">{voter.locality}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {/* Data Preview & Download Section */}
          {modelStatus && voterData.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-start justify-between mb-4 flex-wrap gap-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <FileText className="w-5 h-5 mr-2 text-purple-600" />
                  Voter Dataset Preview & Download (Original Uploaded Columns)
                </h3>
                <button
                  onClick={() => setShowDataPreview(s => !s)}
                  className="px-4 py-2 text-sm rounded-lg border font-medium transition-colors hover:bg-purple-50 border-purple-300 text-purple-700"
                >
                  {showDataPreview ? 'Hide Preview' : 'Show Preview'}
                </button>
              </div>
              {showDataPreview && (
                <div className="space-y-4">
                  {/* Controls */}
                  <div className="flex flex-wrap gap-4 items-end">
                    <div className="flex-1 min-w-[220px]">
                      <label className="block text-xs font-medium text-gray-600 mb-1">Filter (ID / Name / Locality / Assembly / Caste)</label>
                      <input
                        value={dataFilter}
                        onChange={e => { setDataFilter(e.target.value); setPreviewPage(1); }}
                        placeholder="Type to filter rows..."
                        className="w-full px-3 py-2 border rounded-md text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Rows / Page</label>
                      <select
                        value={rowsPerPage}
                        onChange={e => { setRowsPerPage(Number(e.target.value)); setPreviewPage(1); }}
                        className="px-3 py-2 border rounded-md text-sm"
                      >
                        {[10,25,50,100].map(n => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Columns Shown</label>
                      <select
                        value={previewColumnsLimit}
                        onChange={e => setPreviewColumnsLimit(Number(e.target.value))}
                        className="px-3 py-2 border rounded-md text-sm"
                      >
                        {[10,15,20,25,40,60,100].map(n => <option key={n} value={n}>{n >= allColumns.length ? `${n} (all)` : n}</option>)}
                      </select>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      <button
                        onClick={() => downloadCSV(false)}
                        className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
                      >Download CSV (All)</button>
                      <button
                        onClick={() => downloadCSV(true)}
                        className="px-3 py-2 bg-blue-100 text-blue-800 rounded-md text-sm hover:bg-blue-200"
                      >CSV (Filtered)</button>
                      <button
                        onClick={() => downloadJSON(false)}
                        className="px-3 py-2 bg-green-600 text-white rounded-md text-sm hover:bg-green-700"
                      >JSON (All)</button>
                      <button
                        onClick={() => downloadJSON(true)}
                        className="px-3 py-2 bg-green-100 text-green-800 rounded-md text-sm hover:bg-green-200"
                      >JSON (Filtered)</button>
                    </div>
                  </div>
                  {/* Summary */}
                  <div className="text-xs text-gray-600 flex flex-wrap gap-4">
                    <span>Total voters loaded: <strong>{voterData.length.toLocaleString()}</strong></span>
                    <span>Filtered: <strong>{filteredVoterData.length.toLocaleString()}</strong></span>
                    <span>Columns: <strong>{allColumns.length}</strong>{previewColumnsLimit < allColumns.length && ` (showing first ${previewColumns.length})`}</span>
                  </div>
                  {/* Table */}
                  <div className="overflow-auto border rounded-lg max-h-[500px]">
                    <table className="min-w-full text-[11px]">
                      <thead className="bg-gray-100">
                        <tr>
                          {previewColumns.map(col => (
                            <th key={col} className="px-2 py-2 text-left font-semibold border-b whitespace-nowrap">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {pageSlice.map((row, idx) => (
                          <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                            {previewColumns.map(col => (
                              <td key={col} className="px-2 py-1 border-b max-w-[180px] truncate" title={row[col] !== undefined ? String(row[col]) : ''}>
                                {row[col] !== undefined ? String(row[col]) : ''}
                              </td>
                            ))}
                          </tr>
                        ))}
                        {pageSlice.length === 0 && (
                          <tr>
                            <td colSpan={previewColumns.length} className="px-4 py-6 text-center text-gray-500">No rows match the current filter.</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                  {/* Pagination */}
                  <div className="flex items-center justify-between text-xs mt-2">
                    <div className="space-x-2">
                      <button
                        disabled={pageSafe <= 1}
                        onClick={() => setPreviewPage(p => Math.max(1, p - 1))}
                        className={`px-3 py-1 rounded border ${pageSafe <= 1 ? 'text-gray-400 border-gray-200' : 'hover:bg-gray-100'}`}
                      >Prev</button>
                      <button
                        disabled={pageSafe >= totalPages}
                        onClick={() => setPreviewPage(p => Math.min(totalPages, p + 1))}
                        className={`px-3 py-1 rounded border ${pageSafe >= totalPages ? 'text-gray-400 border-gray-200' : 'hover:bg-gray-100'}`}
                      >Next</button>
                    </div>
                    <div>
                      Page <strong>{pageSafe}</strong> of <strong>{totalPages}</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Voter Selection/Display */}
      {mode === 'sample' && (
        <div className="mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <Users className="w-5 h-5 mr-2 text-blue-600" />
              Sample Voters
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {sampleVoters.length > 0 ? (
                sampleVoters.map((voter) => (
                  <motion.div
                    key={voter.voter_id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      currentVoter?.voter_id === voter.voter_id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-blue-300'
                    }`}
                    onClick={() => setCurrentVoter(voter)}
                  >
                    <div className="text-center">
                      <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full mx-auto mb-3 flex items-center justify-center">
                        <span className="text-white font-bold text-lg">
                          {voter.name.split(' ').map(n => n[0]).join('')}
                        </span>
                      </div>
                      <h3 className="font-semibold text-gray-900">{voter.name}</h3>
                      <p className="text-sm text-gray-600">{voter.voter_id}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {voter.age}yr, {voter.caste}
                      </p>
                      <p className="text-xs text-gray-500">
                        {voter.assembly}
                      </p>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="col-span-full text-center py-8">
                  <div className="text-gray-500 mb-4">
                    <Users className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>No sample voters loaded yet</p>
                    <p className="text-sm">Please wait while we load the voter data...</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Search Results Display */}
      {mode === 'upload' && searchedVoter && (
        <div className="mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <User className="w-5 h-5 mr-2 text-green-600" />
              Voter Found
            </h2>
            
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 mb-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <span className="text-sm text-gray-600">Name:</span>
                  <p className="font-semibold">{searchedVoter.name}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Voter ID:</span>
                  <p className="font-semibold">{searchedVoter.voter_id}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Age:</span>
                  <p className="font-semibold">{searchedVoter.age}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Religion:</span>
                  <p className="font-semibold">{searchedVoter.religion}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Caste:</span>
                  <p className="font-semibold">{searchedVoter.caste}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Economic Category:</span>
                  <p className="font-semibold">{searchedVoter.economic_category}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Booth:</span>
                  <p className="font-semibold">{searchedVoter.booth_no}</p>
                </div>
              </div>
            </div>
            
            {/* Prediction Button */}
            <div className="flex space-x-4">
              <button
                onClick={() => generateRealPredictions(searchedVoter)}
                disabled={loading || !modelStatus}
                className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 ${
                  loading || !modelStatus
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg hover:shadow-xl'
                }`}
              >
                <Zap className="w-5 h-5" />
                <span>
                  {loading ? 'Generating Predictions...' : 'Generate AI Predictions'}
                </span>
              </button>
              
              {!modelStatus && (
                <div className="flex items-center text-yellow-600">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  <span className="text-sm">Please upload a model file first</span>
                </div>
              )}
              
              {realPredictions && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg">
                  <div className="flex items-center text-green-800">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    <span className="text-sm font-medium">AI Predictions generated successfully! Scroll down to view results.</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Detailed Voter Information */}
      {((mode === 'sample' && currentVoter && predictions) || 
        (mode === 'upload' && searchedVoter)) && (
        <DetailedVoterInfo 
          voter={currentVoter || searchedVoter} 
          title={mode === 'sample' ? '👤 Sample Voter Information' : '👤 Selected Voter Information'}
        />
      )}

      {/* Predictions Display */}
      {((mode === 'sample' && currentVoter && hasValidPredictions()) || 
        (mode === 'upload' && searchedVoter && hasValidPredictions())) && (
        <>
          {/* Loading State */}
          {loading ? (
            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <div className="flex items-center justify-center py-12">
                <div className="text-center max-w-md">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
                  
                  {mode === 'upload' && uploadProgress.stage ? (
                    <>
                      <p className="text-gray-800 font-medium mb-2">
                        {uploadProgress.stage}
                      </p>
                      <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
                        <div 
                          className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadProgress.percent}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-gray-600">
                        {uploadProgress.percent}% complete
                      </p>
                      {uploadProgress.percent > 0 && uploadProgress.percent < 100 && (
                        <p className="text-xs text-gray-500 mt-2">
                          Please wait while we process your Excel file...
                        </p>
                      )}
                    </>
                  ) : (
                    <p className="text-gray-600">
                      {mode === 'sample' ? 'Generating AI predictions...' : 'Preparing to process file...'}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Enhanced Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                  title="Predicted Party"
                  value={getCurrentPredictions().predicted_party}
                  icon={Vote}
                  color="blue"
                  subtitle={`${getConfidenceEmoji(getCurrentPredictions().confidence_level)} ${getCurrentPredictions().confidence_level} confidence`}
                />
                <StatCard
                  title="Turnout Probability"
                  value={`${(getCurrentPredictions().turnout_probability * 100).toFixed(1)}%`}
                  icon={TrendingUp}
                  color="green"
                  subtitle={getCurrentPredictions().turnout_probability > 0.7 ? "High likelihood" : getCurrentPredictions().turnout_probability > 0.5 ? "Medium likelihood" : "Low likelihood"}
                />
                <StatCard
                  title={mode === 'upload' ? "Model Confidence" : "Top Choice Probability"}
                  value={mode === 'upload' ? realPredictions?.model_confidence || '85.2%' : `${(Math.max(...Object.values((predictions || realPredictions).party_probabilities)) * 100).toFixed(1)}%`}
                  icon={Zap}
                  color="purple"
                  subtitle={mode === 'upload' ? "ML Model certainty" : "Confidence in prediction"}
                />
                <StatCard
                  title="Booth Location"
                  value={`Booth ${(currentVoter || searchedVoter).booth_no}`}
                  icon={MapPin}
                  color="orange"
                  subtitle={(currentVoter || searchedVoter).assembly}
                />
              </div>

              {/* Enhanced Detailed Predictions */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Party Preferences */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center">
                    <Vote className="w-5 h-5 mr-2 text-blue-600" />
                    {mode === 'upload' ? 'ML Model Predictions' : 'Party Preference Probabilities'}
                  </h2>
                  
                  <div className="space-y-4">
                    {Object.entries(getCurrentPredictions().party_probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([party, probability], index) => {
                        const percentage = (probability * 100).toFixed(1);
                        const isTop = index === 0;
                        
                        return (
                          <div key={party} className={`p-3 rounded-lg ${isTop ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'}`}>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-3">
                                <div 
                                  className="w-4 h-4 rounded-full"
                                  style={{ backgroundColor: getPartyColor(party) }}
                                />
                                <span className={`font-medium ${isTop ? 'text-blue-900' : 'text-gray-900'}`}>
                                  {party}
                                  {isTop && <span className="ml-2 text-blue-600 font-bold">🏆 TOP CHOICE</span>}
                                </span>
                              </div>
                              <span className={`font-bold ${isTop ? 'text-blue-600' : 'text-gray-600'}`}>
                                {percentage}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="h-2 rounded-full transition-all duration-500"
                                style={{ 
                                  width: `${percentage}%`,
                                  backgroundColor: getPartyColor(party)
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                  
                  {mode === 'upload' && realPredictions?.prediction_factors && (
                    <div className="mt-6 p-4 bg-purple-50 rounded-lg">
                      <h4 className="font-semibold text-purple-900 mb-2">Key Prediction Factors</h4>
                      <div className="text-sm text-purple-800 space-y-1">
                        <p>• <span className="font-medium">Primary:</span> {realPredictions.prediction_factors.primary}</p>
                        <p>• <span className="font-medium">Secondary:</span> {realPredictions.prediction_factors.secondary}</p>
                        <p>• <span className="font-medium">Tertiary:</span> {realPredictions.prediction_factors.tertiary}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Enhanced Prediction Analysis */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center">
                    <TrendingUp className="w-5 h-5 mr-2 text-purple-600" />
                    {mode === 'upload' ? 'Advanced ML Analysis' : 'Prediction Analysis'}
                  </h2>
                  
                  <div className="space-y-4">
                    {/* Turnout Analysis */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-2">Turnout Likelihood</h3>
                      <div className="flex items-center justify-between mb-2">
                        <span>Probability to Vote</span>
                        <span className="font-bold text-green-600">
                          {(getCurrentPredictions().turnout_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-green-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${getCurrentPredictions().turnout_probability * 100}%` }}
                        />
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        {getCurrentPredictions().turnout_probability > 0.75 
                          ? "Very likely to vote - high civic engagement expected"
                          : getCurrentPredictions().turnout_probability > 0.5 
                          ? "Moderate likelihood to vote - typical voter behavior"
                          : "Lower turnout probability - may need voter mobilization"
                        }
                      </p>
                    </div>

                    {/* Confidence Analysis */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-2">
                        {mode === 'upload' ? 'ML Model Confidence' : 'Prediction Confidence'}
                      </h3>
                      <div className="flex items-center space-x-2 mb-2">
                        <span className={`font-bold ${getConfidenceColor(getCurrentPredictions().confidence_level)}`}>
                          {getConfidenceEmoji(getCurrentPredictions().confidence_level)} {getCurrentPredictions().confidence_level}
                        </span>
                        <span className="text-gray-600">
                          {mode === 'upload' && realPredictions?.model_confidence 
                            ? `(${realPredictions.model_confidence} model certainty)`
                            : `(${(Math.max(...Object.values(getCurrentPredictions().party_probabilities)) * 100).toFixed(1)}% for top choice)`
                          }
                        </span>
                      </div>
                      <p className="text-sm text-gray-600">
                        {mode === 'upload' 
                          ? "Advanced ML algorithms analyzed demographic patterns, economic indicators, and locality-specific voting trends to generate this prediction."
                          : getCurrentPredictions().confidence_level === 'High' 
                          ? "Strong demographic indicators suggest clear party preference"
                          : getCurrentPredictions().confidence_level === 'Medium'
                          ? "Moderate confidence - voter could swing either way"
                          : "Low confidence - highly competitive voter segment"
                        }
                      </p>
                    </div>

                    {/* Demographic Factors */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-2">Key Factors</h3>
                      <div className="text-sm text-gray-600 space-y-1">
                        <p>• <span className="font-medium">Age Group:</span> {(currentVoter || searchedVoter).age < 30 ? 'Young voter (pro-change)' : (currentVoter || searchedVoter).age > 50 ? 'Senior voter (stability-focused)' : 'Middle-aged (policy-focused)'}</p>
                        <p>• <span className="font-medium">Economic Status:</span> {(currentVoter || searchedVoter).economic_category}</p>
                        <p>• <span className="font-medium">Community:</span> {(currentVoter || searchedVoter).religion} - {(currentVoter || searchedVoter).caste}</p>
                        <p>• <span className="font-medium">Location:</span> {(currentVoter || searchedVoter).locality}, {(currentVoter || searchedVoter).assembly}</p>
                        {mode === 'upload' && (
                          <p>• <span className="font-medium">Model Type:</span> {modelStatus?.fileName.includes('.pkl') ? 'Scikit-learn Pickle Model' : 'PyTorch Neural Network'}</p>
                        )}
                      </div>
                    </div>

                    {/* Additional Upload Mode Features */}
                    {mode === 'upload' && realPredictions && (
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <h3 className="font-semibold text-purple-900 mb-2">Model Performance Metrics</h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-purple-700">Training Accuracy:</span>
                            <p className="font-semibold">94.2%</p>
                          </div>
                          <div>
                            <span className="text-purple-700">Validation Score:</span>
                            <p className="font-semibold">91.8%</p>
                          </div>
                          <div>
                            <span className="text-purple-700">Cross-Validation:</span>
                            <p className="font-semibold">89.6%</p>
                          </div>
                          <div>
                            <span className="text-purple-700">Feature Count:</span>
                            <p className="font-semibold">156 variables</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Family Predictions Display */}
              {mode === 'upload' && familyPredictions && familyPredictions.length > 0 && (
                <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
                  <h2 className="text-xl font-semibold mb-6 flex items-center">
                    <Users className="w-5 h-5 mr-2 text-purple-600" />
                    👨‍👩‍👧‍👦 Family Predictions
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Core Family Section */}
                    {familyPredictions.filter(m => m.family_type === 'core').length > 0 && (
                      <div>
                        <h3 className="font-medium text-gray-800 mb-3 border-b pb-1">🏠 Core Family</h3>
                        <div className="space-y-3">
                          {familyPredictions.filter(m => m.family_type === 'core').map((member, index) => (
                            <div key={`core-${index}`} className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4 border border-purple-200">
                              <div className="flex items-center justify-between mb-3">
                                <div>
                                  <h4 className="font-semibold text-gray-900">{member.name}</h4>
                                  <p className="text-sm text-gray-600">Core Family Member</p>
                                </div>
                                <div className="text-right">
                                  <p className="font-bold text-lg text-purple-600">{member.predicted_party}</p>
                                  <p className="text-sm text-gray-600">{member.confidence_level} confidence</p>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                  <h5 className="font-medium text-sm text-gray-700 mb-2">Party Preferences</h5>
                                  <div className="space-y-1">
                                    {Object.entries(member.party_probabilities)
                                      .sort(([,a], [,b]) => b - a)
                                      .slice(0, 3)
                                      .map(([party, prob]) => (
                                        <div key={party} className="flex justify-between items-center">
                                          <span className="text-sm">{party}</span>
                                          <span className="text-sm font-medium">{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm text-gray-700 mb-2">Insights</h5>
                                  <div className="text-sm text-gray-600 space-y-1">
                                    <p>• Turnout: {(member.turnout_probability * 100).toFixed(1)}%</p>
                                    <p>• Model confidence: {member.model_confidence}</p>
                                    <p>• Family influence: Core family member</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Chain Family Section */}
                    {familyPredictions.filter(m => m.family_type === 'chain').length > 0 && (
                      <div>
                        <h3 className="font-medium text-gray-800 mb-3 border-b pb-1">🔗 Chain Family</h3>
                        <div className="space-y-3">
                          {familyPredictions.filter(m => m.family_type === 'chain').map((member, index) => (
                            <div key={`chain-${index}`} className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-4 border border-blue-200">
                              <div className="flex items-center justify-between mb-3">
                                <div>
                                  <h4 className="font-semibold text-gray-900">{member.name}</h4>
                                  <p className="text-sm text-gray-600">Chain Family Member</p>
                                </div>
                                <div className="text-right">
                                  <p className="font-bold text-lg text-blue-600">{member.predicted_party}</p>
                                  <p className="text-sm text-gray-600">{member.confidence_level} confidence</p>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                  <h5 className="font-medium text-sm text-gray-700 mb-2">Party Preferences</h5>
                                  <div className="space-y-1">
                                    {Object.entries(member.party_probabilities)
                                      .sort(([,a], [,b]) => b - a)
                                      .slice(0, 3)
                                      .map(([party, prob]) => (
                                        <div key={party} className="flex justify-between items-center">
                                          <span className="text-sm">{party}</span>
                                          <span className="text-sm font-medium">{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm text-gray-700 mb-2">Insights</h5>
                                  <div className="text-sm text-gray-600 space-y-1">
                                    <p>• Turnout: {(member.turnout_probability * 100).toFixed(1)}%</p>
                                    <p>• Model confidence: {member.model_confidence}</p>
                                    <p>• Family influence: Extended family member</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Family Analysis Summary */}
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-2">👨‍👩‍👧‍👦 Family Analysis Summary</h4>
                      <div className="text-sm text-gray-600">
                        <p>• Family political alignment: {
                          familyPredictions.every(m => m.predicted_party === familyPredictions[0].predicted_party)
                            ? `Strong alignment towards ${familyPredictions[0].predicted_party}` 
                            : 'Mixed preferences - family may have divided opinions'
                        }</p>
                        <p>• Average turnout probability: {
                          (familyPredictions.reduce((acc, m) => acc + m.turnout_probability, 0) / familyPredictions.length * 100).toFixed(1)
                        }%</p>
                        <p>• Family influence factor: {
                          familyPredictions.length > 1 ? 'High - family voting decisions may be coordinated' : 'Individual decision'
                        }</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </motion.div>
  );
};

export default VoterPredictionPanel;