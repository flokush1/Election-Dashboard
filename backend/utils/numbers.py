import math
import re

import numpy as np
import pandas as pd


def to_float_safe(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float, np.number)):
        try:
            v = float(val)
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
        return default
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            v = float(s[:-1]) / 100.0
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    try:
        return float(m.group()) if m else default
    except Exception:
        return default


def safe_int(val, default=0):
    try:
        if val is None:
            return default
        s = str(val).strip()
        if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
            return default
        return int(float(s))
    except Exception:
        return default


def get_any(row_or_dict, *names, default=None):
    if isinstance(row_or_dict, pd.Series):
        data = row_or_dict.to_dict()
    else:
        data = dict(row_or_dict)
    for n in names:
        if n in data and pd.notna(data[n]):
            return data[n]
    lower = {str(k).lower(): v for k, v in data.items()}
    for n in names:
        key = str(n).lower()
        if key in lower and pd.notna(lower[key]):
            return lower[key]
    return default
