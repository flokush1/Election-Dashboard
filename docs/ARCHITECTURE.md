# Architecture

The app is a React/Vite dashboard with a Flask API.

- `src/app` application shell, routes, error boundary
- `src/features/hierarchy` parliament/assembly/ward navigation
- `src/features/booth` booth stats, vote allocation, booth page
- `src/features/predictions` voter upload/search/predict
- `src/features/maps` interactive and booth maps
- `src/shared/api` HTTP client
- `backend/api` Flask blueprints
- `backend/services` use cases
- `backend/domain` name matching and voter normalization
- `backend/ml` Streamlit-free predictor
- `backend/repositories` Excel/CSV/file access

Startup remains `npm run dev` and `python model_api.py`.
