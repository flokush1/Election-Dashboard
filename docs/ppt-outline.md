# Delhi Election Dashboard – Slide Outline

1. Title & Purpose
- What: Interactive dashboard for Delhi elections.
- Why: Insight across Parliament → Assembly → Ward → Booth → Building/Voter.
- Scope: Winner maps, demographics, economics, predictions.

2. Executive Summary
- Outcome: End-to-end election view with drilldowns.
- Stack: React + Leaflet + Recharts + Tailwind; Flask backend.
- Data: Electoral parts JSON, boundaries GeoJSON, detailed booths, predictions.

3. Architecture Overview
- Frontend: SPA in React 18 + Vite; Tailwind styling.
- Mapping: `react-leaflet` for choropleths + overlays.
- Backend: Flask (`model_api.py`) with previews, search, predictions.
- Flow: Public JSON/GeoJSON → normalized joins → aggregated views.

4. Data Model Overview
- Granularity: Part (polling segment) is base unit.
- Layers: Parliament/Assembly/Ward/Booth boundaries GeoJSON.
- Specials: New Delhi Booth 1 buildings; RK Puram Booth 17 plots.

5. Electoral Part Dataset Schema
- File: `public/data/electoral-data.json` (array of parts).
- Identifiers: `PartNo`, `AssemblyNo`, `AssemblyName`, `Ward No.`, `Ward Name`.
- Demographics: Age ratios, gender ratios, `TotalPop`.
- Religion/Caste: `Religion_*_Ratio`, `Caste_*_Ratio`.
- Economics: `economic_category`, land/construction rates.
- Voting: Party ratios, `Total_Polled`, `Winner`, `Margin`.

6. GeoJSON Layers
- Wards: `ward-boundaries.geojson` → `Ward_No`, `WardName`, `AC_No`, `AC_Name`.
- Assemblies: `assembly-boundaries.geojson` (ACS boundaries).
- Parliament: `parliament-boundaries.geojson` (PCS boundaries).
- Detailed: `New_Delhi_Booth_Buildings.geojson`, `New_Delhi_Booth_Data.geojson`, `RKPuram_Booth_17_Boundary.geojson`, `RKPuram_Booth_17_Plots_With_Predictions.geojson`.

7. Name Normalization & Matching
- Utilities: `canonicalWardKey`, `normalizeWardDisplay`.
- Purpose: Stable joins between data and GeoJSON despite format differences.
- Result: Correct coloring/tooltips across all maps.

8. Navigation & Granularity
- Hierarchy: Parliament → Assembly → Ward → Booth → Building/Voter.
- UX: Click-through maps/lists; legends explaining party colors.
- Fallbacks: Graceful handling when detailed layers are missing.

9. Parliament Level
- View: Assembly breakdown within the parliamentary seat.
- Metrics: Winner distribution, turnout, party share.
- API: `GET /api/parliament-data-preview`.

10. Assembly Level
- View: Ward map colored by winner; charts for composition.
- Metrics: Party shares, demographics, economics.
- API: `GET /api/assembly-data-preview?assembly=<name>`.

11. Ward Level
- View: Booth list and compact charts; optional booth map.
- Metrics: Booth winners, margins, local demographics.
- Source: Aggregation from part-level records.

12. Booth Level
- View: Recomputed votes (largest remainder) + segmentation (Core/Leaning/Swing).
- Predictions: Fetch voter predictions when available.
- Component: `BoothLevel.jsx` with detail panels.

13. Detailed Booth – New Delhi Booth 1
- Layers: Buildings multipolygons with `Booth_No`, `A_CNST_NM`.
- Map: Overlay with predicted or contextual coloring.
- Component: `BoothDetailMap.jsx` special-case support.

14. Detailed Booth – RK Puram Booth 17
- Boundary + plots with predictions.
- Plot props: `voter_count`, `avg_turnout_prob`, `avg_prob_*`, `predicted_winner`, `voters[]`.
- Use: Plot popups for winner + voters.

15. Voter Predictions
- Endpoints: `/api/predict`, `/api/predict-batch`, `/api/predict-family`, `/api/voter-prediction/<id>`.
- Input: Voter features; model uploaded via `/api/upload-model`.
- UI: `VoterPredictionCard.jsx` shows per-party probabilities.

16. Charts & Insights
- Party share bars, age group, demographics.
- Economics: `economic_category`, land/construction cost summaries.
- Libraries: Recharts + Tailwind styling.

17. Aggregation & Votes
- Aggregation: `src/shared/dataProcessor.js` for multi-level rollups.
- Votes: Largest remainder method to avoid rounding bias.
- Matching: Robust to partial data presence.

18. Backend APIs & Uploads
- Health: `GET /api/health`.
- Uploads: `/api/upload-voter-data` (Excel), `/api/upload-model` (pickle).
- Search: `/api/search-voter`, `/api/available-voters`, `/api/debug-voter`.

19. Performance & UX
- Server-side previews reduce client load.
- Conditional loading for heavy GeoJSON layers.
- Special-case clipping for detailed booths.

20. Troubleshooting & Fallbacks
- Guide: `VOTER_SEARCH_TROUBLESHOOTING.md`.
- Scripts: `test_backend.ps1`, `restart_server.ps1`.
- Fallbacks: Name normalization and safe defaults.

21. Limitations & Next Steps
- Detail limited to specific booths/buildings.
- Predictions depend on uploaded model/data quality.
- Next: Expand detailed layers; refine economic overlays; add more benchmarks.
