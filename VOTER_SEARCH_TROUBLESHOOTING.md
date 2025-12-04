# Voter Search Troubleshooting Guide

## Problem: Voter IDs Not Found / Data Not Loading

This guide helps you diagnose and fix issues with the voter prediction search functionality.

---

## Quick Diagnostic Checklist

### ✅ Step 1: Verify Backend is Running

**Check if Python API server is running on port 5000:**

```powershell
# Check if Python process is running
Get-Process python -ErrorAction SilentlyContinue

# Check if port 5000 is listening
netstat -ano | findstr :5000
```

**If NOT running, start it:**

```powershell
cd "c:\Users\kushp\Downloads\testing\Delhi election Model\New Delhi Parliamentary\delhi-election-dashboard"
python model_api.py
```

You should see output like:
```
✅ scikit-learn: 1.x.x
🔍 Checking dependencies...
 * Running on http://127.0.0.1:5000
```

---

### ✅ Step 2: Verify Frontend is Running

```powershell
# Should already be running npm run dev
# Check browser at: http://localhost:3000
```

---

### ✅ Step 3: Check Browser Console

**Open Developer Tools (F12) and look for:**

1. **Network tab**: Check if API calls to `/api/upload-voter-data` and `/api/search-voter` return 200 OK
2. **Console tab**: Look for these debug messages:

```
✅ ===== DATA UPLOAD SUCCESS =====
✅ Total voters loaded: [NUMBER]
✅ Sample voter IDs: [Array of IDs]
```

If you DON'T see these messages, data upload failed!

---

## Common Issues & Solutions

### Issue 1: "No Data Uploaded" or Empty Results

**Symptoms:**
- Search says "0 voters loaded"
- No sample IDs shown
- Alert popup doesn't appear after upload

**Causes:**
1. **Excel file format issues**
2. **Column name mismatch**
3. **Backend not storing data**

**Solution:**

#### A. Check Excel File Structure

Your Excel file MUST have at least these columns (case-insensitive):
- `Voter ID` or `voter_id` or `EPIC` or `ID` (required for search)
- `Name` or `name` or `Voter Name`
- `Age` or `age`
- Other columns as needed

**Open your Excel and verify:**
```
| Voter ID    | Name         | Age | Gender | ...
|-------------|--------------|-----|--------|...
| ABC1234567  | John Doe     | 35  | Male   |...
| XYZ7654321  | Jane Smith   | 42  | Female |...
```

#### B. Check Backend Logs

Look at the Python terminal where `model_api.py` is running:

**Successful upload shows:**
```
✅ Stored [NUMBER] voters in global cache for search
```

**Failed upload shows:**
```
❌ Upload error: ...
```

#### C. Try Server-Side Processing

If your file is under 50MB, the backend should process it automatically. Check console for:
```
✅ Server processed X/Y rows successfully
```

If you see fallback messages like:
```
⚠️ Server processing failed, falling back to client-side
```

The backend had an issue. Check Python terminal for errors.

---

### Issue 2: "Voter ID Not Found" Even Though Data Loaded

**Symptoms:**
- Alert shows "✅ Successfully loaded X voters"
- Sample IDs are displayed
- But searching for those exact IDs fails

**Debug Steps:**

#### 1. Open Browser Console and try searching

You should see:
```
🔍 ===== VOTER SEARCH DEBUG =====
🔍 Searching for: [YOUR_ID]
🔍 Total voters in local data: [NUMBER]
🔍 Sample local voter IDs: [Array]
```

#### 2. Compare your search input with sample IDs

**Common mistakes:**
- **Extra spaces**: "ABC123 " vs "ABC123"
- **Case sensitivity**: "abc123" vs "ABC123" (should work now with new code)
- **Special characters**: "ABC-123" vs "ABC123"

**Fix: Use the improved search:**
The updated code now tries 3 strategies:
1. Exact match (case-insensitive)
2. Partial match
3. Normalized match (removes special chars)

#### 3. Click on Sample IDs

Instead of typing, **CLICK** on the blue voter ID badges shown below the upload section. This auto-fills the search box with the exact ID.

---

### Issue 3: Backend Search Fails

**Symptoms:**
```
❌ Backend search error: Failed to fetch
```

**Causes:**
1. Backend not running
2. Port 5000 blocked
3. CORS issue

**Solutions:**

#### A. Restart Backend
```powershell
# Stop any existing Python processes
Get-Process python | Stop-Process -Force

# Start fresh
cd "c:\Users\kushp\Downloads\testing\Delhi election Model\New Delhi Parliamentary\delhi-election-dashboard"
python model_api.py
```

#### B. Check Vite Proxy

The frontend proxies `/api` requests to `http://localhost:5000`. 

Verify `vite.config.js` has:
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:5000',
      changeOrigin: true
    }
  }
}
```

#### C. Test Backend Directly

Open a new PowerShell and test:
```powershell
curl http://localhost:5000/api/health
```

Should return:
```json
{"status":"healthy","model_loaded":false}
```

---

### Issue 4: Data Upload Takes Forever

**For large files (>100MB):**

#### Progress Monitoring

Watch the upload progress bar. It should show stages like:
```
Processing large file, please wait... 10%
Reading file... 20%
Processing voters: 5000/10000 (50%) 50%
Finalizing voter data... 85%
Completed successfully! 100%
```

**If stuck:**

1. **Check file size:**
   ```powershell
   Get-Item "path\to\your\voter_file.xlsx" | Select-Object Length
   ```

2. **Files over 300MB**: Split into smaller chunks
3. **Files 50-300MB**: Be patient, can take 2-5 minutes
4. **Files under 50MB**: Should complete in <30 seconds

---

## Testing Your Fix

### 1. Upload Test Data

Create a small test Excel file:

```
| Voter_ID  | Name      | Age | Gender |
|-----------|-----------|-----|--------|
| TEST001   | Test One  | 30  | Male   |
| TEST002   | Test Two  | 35  | Female |
```

Save as `test_voters.xlsx`

### 2. Upload and Search

1. Go to "Electoral Individual Voter Prediction Dashboard"
2. Upload your test file
3. Wait for alert: "✅ Successfully loaded 2 voters!"
4. You should see sample IDs: `TEST001`, `TEST002`
5. Click on `TEST001` badge - it auto-fills search
6. Click "Search" button
7. Should show: "✅ Found voter locally: TEST001 Test One"

### 3. Verify in Console

Check browser console (F12):
```
✅ ===== DATA UPLOAD SUCCESS =====
✅ Total voters loaded: 2
✅ Sample voter IDs: ["TEST001", "TEST002"]

🔍 ===== VOTER SEARCH DEBUG =====
🔍 Searching for: TEST001
🔍 Total voters in local data: 2
🔍 Strategy 1: Exact match (case-insensitive)
✅ Found voter locally: TEST001 Test One
```

---

## Advanced Debugging

### Check Uploaded Data in Memory

Open browser console and type:
```javascript
// Get React component state (if using React DevTools)
// Or check these console logs after upload
```

Look for the debug logs showing:
```
✅ First voter example: {voter_id: "...", name: "...", ...}
✅ Voter ID column detected: ...
```

### Check Backend Data Cache

In Python terminal, after upload, check:
```python
# The backend prints:
✅ Stored [NUMBER] voters in global cache for search

# And when searching:
🔍 Searching for voter ID: [ID]
✅ Found voter: [NAME] (ID: [ID])
# OR
❌ Voter ID [ID] not found. Available IDs sample: [...]
```

---

## Still Not Working?

### Collect Full Debug Info

1. **Browser Console Logs** (F12 → Console tab → Copy all)
2. **Network Tab** (F12 → Network → Filter: api → Right-click → Copy all as HAR)
3. **Python Terminal Output** (Copy last 50 lines)
4. **Your Excel file structure** (Screenshot of first few rows with columns visible)

### Common Root Causes

1. ✗ **Backend not running** → Start `python model_api.py`
2. ✗ **Wrong port** → Backend must be port 5000, frontend port 3000
3. ✗ **Excel has no voter ID column** → Add `Voter_ID` or `EPIC` column
4. ✗ **Firewall blocking localhost** → Allow port 5000
5. ✗ **Browser cache** → Hard refresh (Ctrl+Shift+R)
6. ✗ **CORS issue** → Check `flask_cors` is installed: `pip install flask-cors`

---

## Success Indicators

When everything works correctly:

1. ✅ Backend shows: `✅ Stored [N] voters in global cache`
2. ✅ Browser alert: `✅ Successfully loaded [N] voters!`
3. ✅ Sample IDs visible and clickable
4. ✅ Search finds voters with debug logs showing each strategy
5. ✅ Predictions generate and display correctly

---

## Contact Support

If still stuck, provide:
- Browser console logs
- Python terminal output
- Excel file sample (first 10 rows)
- Screenshots of error messages
