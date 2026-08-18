from functools import lru_cache
from threading import Lock

import pandas as pd

from backend.config import config


_excel_lock = Lock()


@lru_cache(maxsize=4)
def _read_excel_cached(path, modified_time):
    del modified_time
    return pd.read_excel(path)


def resolve_excel_path():
    candidates = [
        config.EXCEL_PATH,
        config.PROJECT_ROOT / 'NewDelhi_Parliamentary_Data.xlsx',
        config.PRIVATE_DATA_DIR / 'raw' / 'NewDelhi_Parliamentary_Data.xlsx',
    ]
    for path in candidates:
        if path.exists():
            return path
    return config.EXCEL_PATH


def excel_available():
    return resolve_excel_path().exists()


def read_parliamentary_excel(nrows=None):
    path = resolve_excel_path()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if nrows:
        return pd.read_excel(path, nrows=nrows)
    with _excel_lock:
        return _read_excel_cached(str(path), path.stat().st_mtime).copy()
