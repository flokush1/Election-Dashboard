# Speaker Notes – Delhi Election Dashboard

Slide 1 – Title & Purpose
- Introduce the dashboard as a multi-level decision and insight tool for Delhi elections.
- Emphasize drilldown from Parliament to individual buildings/voters where available.
- Call out that content is grounded in `public/data/*` and `src/components/**`.

Slide 2 – Executive Summary
- Summarize outcomes: winner maps, demographics/economics, segmentation, and predictions.
- Mention stack briefly: React + Leaflet + Recharts + Tailwind; Flask backend.
- Set audience expectation: we’ll walk level-by-level with real schema fields and endpoints.

Slide 3 – Architecture Overview
- Data enters as JSON/GeoJSON (and optional Excel uploads) and is normalized/matched.
- Frontend (`InteractiveMap.jsx`, level components) renders maps and charts; backend (`model_api.py`) serves previews, search, and predictions.
- Name normalization in `src/shared/utils.js` ensures robust joins.

Slide 4 – Data Model Overview
- Base unit is the polling Part in `electoral-data.json`.
- Boundaries: wards/assemblies/parliament shapes; detailed booths for New Delhi Booth 1 and RK Puram Booth 17.
- Prediction CSVs and detailed plots enrich lower levels.

Slide 5 – Electoral Part Dataset Schema
- Show exact fields: identifiers (Part/Assembly/Ward), demographics (age/gender), religion/caste ratios, economics, and voting metrics.
- Highlight `Winner` and `Margin` as used for choropleths and rollups.
- Note scales: ratios are fractional shares per part; totals used for weighting.

Slide 6 – GeoJSON Layers
- Wards: `Ward_No`, `WardName` (with `AC_No`, `AC_Name`) from `ward-boundaries.geojson`.
- Assembly and Parliament: ACS/PCS feature collections for upper-level maps.
- Detailed booth layers: buildings for New Delhi Booth 1; plots with predictions for RK Puram Booth 17.

Slide 7 – Name Normalization & Matching
- Explain `canonicalWardKey` and `normalizeWardDisplay` to mitigate spacing/case/abbreviation mismatches.
- Stress this is why the map correctly binds to data even with messy inputs.

Slide 8 – Navigation & Granularity
- Users start at Parliament, then drill to Assembly, Ward, Booth, and sometimes Buildings/Voters.
- Legends clarify party colors; tooltips/popups show feature properties and winners.
- Missing layers handled with graceful fallbacks (lists/charts still render).

Slide 9 – Parliament Level
- Uses `GET /api/parliament-data-preview` to load aggregated snapshots.
- Assembly-level map displays winners; charts show party shares and turnout context.
- Aligns with color palette from `src/shared/utils.js`.

Slide 10 – Assembly Level
- Ward choropleth by winner; charts for demographics/economics.
- `GET /api/assembly-data-preview?assembly=<name>` returns server-side rollups.
- Demonstrate a sample assembly (e.g., Rajinder Nagar) tying to ward features.

Slide 11 – Ward Level
- Booth list with winner/margin stats; optional booth map when detailed geometry exists.
- Aggregations computed from parts matching the ward; normalized keys ensure consistency.

Slide 12 – Booth Level
- Booth view recomputes votes via largest remainder method to avoid rounding bias.
- Segments voters/areas into Core, Leaning, Swing buckets based on probabilities.
- Predictions fetched if available; otherwise the UI degrades gracefully.

Slide 13 – Detailed Booth – New Delhi Booth 1
- `New_Delhi_Booth_Buildings.geojson` & `New_Delhi_Booth_Data.geojson` provide building geometries and booth context.
- Popups show building-level attributes; color can reflect predicted winner or category.

Slide 14 – Detailed Booth – RK Puram Booth 17
- `RKPuram_Booth_17_Boundary.geojson` and `RKPuram_Booth_17_Plots_With_Predictions.geojson` drive this rich layer.
- Each plot includes `voter_count`, `avg_turnout_prob`, per-party averages, `predicted_winner`, and `voters[]` with detailed features.
- Use this slide to showcase a real popup example (winner and a couple of voters).

Slide 15 – Voter Predictions
- Endpoints: per voter (`/api/voter-prediction/<id>`), batch (`/api/predict-batch`), family (`/api/predict-family`).
- Models are uploaded to `/api/upload-model` and wrapped by the backend for consistent features.
- UI card (`VoterPredictionCard.jsx`) renders probabilities and turnout.

Slide 16 – Charts & Insights
- Party share bars, age group distributions, demographics panels.
- Economics: `economic_category`, `land_rate_per_sqm`, `construction_cost_per_sqm` summarized per area.
- Keep visuals simple and consistent with the color palette.

Slide 17 – Aggregation & Votes
- `src/shared/dataProcessor.js` handles rollups across levels with correct weighting.
- Largest remainder for recomputing votes at booth level prevents drift from rounding.
- Mention how missing data is handled to maintain stability.

Slide 18 – Backend APIs & Uploads
- Quick map: Health, previews, search, uploads, predictions, debug endpoints from `model_api.py`.
- Data upload (`/api/upload-voter-data`) normalizes column names and caches frames for later use.
- `POST /api/debug-voter` traces transformations for a specific voter.

Slide 19 – Performance & UX
- Server-side previews keep initial payloads small.
- Conditional/optional layers (e.g., detailed booths) avoid heavy initial loads.
- Special-case clipping and selective rendering in `BoothDetailMap.jsx` ensure responsiveness.

Slide 20 – Troubleshooting & Fallbacks
- Refer to `VOTER_SEARCH_TROUBLESHOOTING.md` for search/prediction issues.
- Use `test_backend.ps1` and `restart_server.ps1` locally when iterating.
- Fallbacks: default mappings and normalized names prevent blank maps.

Slide 21 – Limitations & Next Steps
- Detailed building/plot views are limited to select areas (e.g., RK Puram 17).
- Prediction quality depends on uploaded model and feature coverage.
- Roadmap: expand detailed geospatial coverage; refine economics overlays; add more assemblies’ predictions.
