#!/usr/bin/env python3
"""Compatibility entrypoint. The Flask app now lives in backend.app_factory."""

from backend.app_factory import create_app
from backend.config import config

app = create_app()

if __name__ == '__main__':
    print("Starting Election Dashboard API Server")
    print(f"Host: {config.FLASK_HOST}:{config.FLASK_PORT}")
    print("Health Check: GET  /api/health")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
