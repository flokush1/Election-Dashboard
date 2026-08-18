# Commands and How to Use the Dashboard

This file is the practical guide for running and using the New Delhi Election Dashboard.

The app has two processes:

- **Frontend** (React / Vite): `http://localhost:3000`
- **Backend** (Flask API): `http://127.0.0.1:5000`

Keep both running while you use the dashboard.

---

## 1. Prerequisites

- Node.js 18 or newer (`node -v`)
- npm (`npm -v`)
- Python 3.10 or newer (`python --version`)
- Git

On Windows, run the commands below in **PowerShell** from the project root:

```powershell
cd "C:\Users\kushp\Downloads\Kush_Data\Kush Data\Voter Management\Delhi election Model\New Delhi Parliamentary\delhi-election-dashboard"
```

---

## 2. First-time setup

### Frontend

```powershell
npm install
```

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Optional environment file

```powershell
copy .env.example .env
```

Default ports are already correct:

- Frontend: `3000`
- API: `5000`

---

## 3. Start the dashboard (daily use)

Open **two terminals** in the project root.

### Terminal 1 — API

```powershell
.\.venv\Scripts\Activate.ps1
python model_api.py
```

Wait until you see the server start on port `5000`.

### Terminal 2 — UI

```powershell
npm run dev
```

Then open:

**http://localhost:3000**

Check that the API is healthy:

```powershell
Invoke-WebRequest http://127.0.0.1:5000/api/health
```

You can also visit `http://127.0.0.1:5000/api/health` in the browser. You should see `"status": "healthy"`.

---

## 4. How to use the dashboard

### Parliament home (`/`)

This is the New Delhi Parliamentary overview.

- Review vote totals, booths, and party performance.
- Click an assembly on the **map** or use **Jump to Assembly**.
- Click **AI Voter Predictions** to open the ML upload/search screen.

### Assembly (`/assembly/:name`)

- Use the dropdowns to switch assembly or jump into a ward.
- Click a ward on the map to go deeper.
- Preview rows come from the parliamentary Excel file if it is available locally.

### Ward (`/assembly/:name/ward/:name`)

- See booths inside the selected ward.
- Click a booth on the map or from the list.

### Booth (`/assembly/:name/ward/:name/booth/:number`)

- See booth vote share, demographics, and predicted voters.
- If detailed map data exists (for example New Delhi booth 103 or R.K. Puram booth 17), the plot map appears.
- Click a building/plot to open the detail panel.
- Use the voter dropdown to inspect one voter’s prediction.

### AI voter predictions (`/predictions`)

1. Confirm **ML API: Connected** in the header.
2. Upload a `.pkl` or `.pth` model.
3. Upload a voter Excel file (`.xlsx` / `.xls`).
4. Search a voter ID.
5. Run **Predict** for that voter.
6. Family predictions appear when the Excel file includes family IDs.

Browser Back works across these screens because the hierarchy is now in the URL.

---

## 5. Private data (optional, for full booth/ML features)

Git does **not** include voter-level CSVs, Excel rolls, or model files.

Place local files here, or keep them at the repo root (the API still finds both):

```text
data/private/predictions/     predictions_new_delhi.csv, predictions_r_k_puram.csv
data/private/raw/             NewDelhi_Parliamentary_Data.xlsx
data/private/models/          .pkl / .pth model files
data/private/voter_rolls/     VoterID assembly Excel files
```

Public map/aggregate files already live in `public/data/` and are enough for the parliament/assembly/ward maps.

---

## 6. Other useful commands

### Stop servers

In each terminal: `Ctrl+C`

### Frontend production build

```powershell
npm run build
npm run preview
```

`preview` serves the built files. The API must still be running on port `5000` if you want live booth/prediction data.

### Tests

```powershell
npm test
python -m pytest
```

### Lint frontend

```powershell
npm run lint
```

### Docker (optional)

From the repo root:

```powershell
docker compose -f deploy/docker-compose.yml up --build
```

This serves the UI on port `80` and the API on port `5000`. Mount private data as described in `docs/DEPLOYMENT.md`.

---

## 7. Common API endpoints

These are used by the UI. You can also call them directly.

| What | Command |
|------|---------|
| Health | `GET http://127.0.0.1:5000/api/health` |
| Parliament preview | `GET http://127.0.0.1:5000/api/parliament-data-preview?limit=15` |
| Assembly preview | `GET http://127.0.0.1:5000/api/assembly-data-preview?assembly=NEW%20DELHI` |
| Booth stats | `GET http://127.0.0.1:5000/api/booth-statistics/R%20K%20PURAM/17` |
| Booth voters | `GET http://127.0.0.1:5000/api/voter-predictions/R%20K%20PURAM/17` |
| One voter | `GET http://127.0.0.1:5000/api/voter-prediction/<voter_id>` |
| Upload model | `POST /api/upload-model` (form field `model`) |
| Upload voters | `POST /api/upload-voter-data` (form field `file`) |
| Search voter | `POST /api/search-voter` JSON `{ "voter_id": "..." }` |
| Predict | `POST /api/predict` JSON voter object |

---

## 8. Troubleshooting

| Problem | What to do |
|---------|------------|
| UI spinner never finishes | Confirm `npm run dev` is running and `public/data/electoral-data.json` exists. |
| ML API: Disconnected | Start `python model_api.py` first, then refresh `http://localhost:3000/predictions`. |
| Booth predictions empty | Put the matching `predictions_*.csv` in `data/private/predictions` or the repo root, then restart the API. |
| Excel preview 404 | Place `NewDelhi_Parliamentary_Data.xlsx` in `data/private/raw` or the repo root. The rest of the dashboard still works. |
| Map tiles missing | Needs internet for OpenStreetMap tiles. |
| Port 3000 or 5000 already in use | Stop the old terminal with `Ctrl+C`, or change `FLASK_PORT` in `.env`. |
| `python model_api.py` import error | Run it from the **project root**, with the venv activated, after `pip install -r requirements.txt`. |
| PowerShell cannot activate venv | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again. |

---

## 9. More documentation

- `docs/SETUP.md` — short setup
- `docs/ARCHITECTURE.md` — folder layout
- `docs/DATA_LIFECYCLE.md` — public vs private data
- `docs/DEPLOYMENT.md` — production / Docker
