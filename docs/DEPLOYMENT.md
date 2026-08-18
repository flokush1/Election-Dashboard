# Production

Use `deploy/docker-compose.yml` to run nginx for the Vite build and gunicorn for Flask.

- Disable debug endpoints with `EXPOSE_DEBUG_ENDPOINTS=false`
- Restrict `CORS_ORIGINS` to the public frontend origin
- Mount `data/private` read-only into the API container
- Keep voter-level CSVs off public remotes
