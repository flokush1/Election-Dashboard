import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env(name, default=None):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return value


def _env_bool(name, default=False):
    value = _env(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = Path(_env('DATA_ROOT', str(PROJECT_ROOT)))
    PRIVATE_DATA_DIR = Path(_env('PRIVATE_DATA_DIR', str(DATA_ROOT / 'data' / 'private')))
    PUBLIC_DATA_DIR = Path(_env('PUBLIC_DATA_DIR', str(PROJECT_ROOT / 'public' / 'data')))
    PREDICTIONS_DIR = Path(_env('PREDICTIONS_DIR', str(PRIVATE_DATA_DIR / 'predictions')))
    MODELS_DIR = Path(_env('MODELS_DIR', str(PRIVATE_DATA_DIR / 'models')))
    VOTER_ROLLS_DIR = Path(_env('VOTER_ROLLS_DIR', str(PRIVATE_DATA_DIR / 'voter_rolls')))
    EXCEL_PATH = Path(_env(
        'EXCEL_DATA_PATH',
        str(PRIVATE_DATA_DIR / 'raw' / 'NewDelhi_Parliamentary_Data.xlsx')
    ))
    FLASK_HOST = _env('FLASK_HOST', '127.0.0.1')
    FLASK_PORT = _env_int('FLASK_PORT', 5000)
    FLASK_DEBUG = _env_bool('FLASK_DEBUG', False)
    FRONTEND_URL = _env('FRONTEND_URL', 'http://localhost:3000')
    CORS_ORIGINS = [origin.strip() for origin in _env('CORS_ORIGINS', FRONTEND_URL).split(',') if origin.strip()]
    ENV = _env('FLASK_ENV', 'development')
    MAX_EXCEL_UPLOAD_MB = _env_int('MAX_EXCEL_UPLOAD_MB', 50)
    MAX_MODEL_UPLOAD_MB = _env_int('MAX_MODEL_UPLOAD_MB', 200)
    LOG_LEVEL = _env('LOG_LEVEL', 'INFO')
    EXPOSE_DEBUG_ENDPOINTS = _env_bool('EXPOSE_DEBUG_ENDPOINTS', ENV != 'production')
    INCLUDE_TRACEBACKS = _env_bool('INCLUDE_TRACEBACKS', ENV != 'production')

    @property
    def is_production(self):
        return self.ENV == 'production'


config = Config()
