# Data lifecycle

## Public vs private

- `public/data` is the only dataset shipped with the dashboard UI. It contains aggregated booth statistics and boundary GeoJSON. It must not include voter IDs, names, or addresses.
- `data/private` is local/server storage for voter rolls, prediction CSVs, Excel source files, and ML artifacts. Do not commit these files.

## Folders

```
data/private/raw/            Parliamentary Excel and source plot files
data/private/predictions/    predictions_*.csv
data/private/models/         .pkl / .pth checkpoints
data/private/voter_rolls/    VoterID assembly Excel files
public/data/                 electoral-data.json, summary-stats.json, boundaries
```

## Flow

1. Place raw Excel/models in `data/private`.
2. Run ETL scripts from `scripts/etl` to generate processed predictions and public aggregates.
3. Copy non-PII aggregates into `public/data`.
4. Flask reads private files through `DATA_ROOT` / `PREDICTIONS_DIR` environment variables.

## Git history

Tracked prediction CSVs were removed from the working tree index going forward. History rewrite is a separate, reviewed operation because it affects clones and remotes.
