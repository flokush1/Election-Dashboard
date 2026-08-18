import math
import re


def norm_key(s):
    try:
        return re.sub(r"[^0-9a-z]+", "_", str(s).strip().lower()).strip("_")
    except Exception:
        return str(s)


def build_colmap(df):
    colmap = {}
    for c in df.columns:
        colmap[norm_key(c)] = c
    return colmap


def find_col(colmap, aliases):
    for a in aliases:
        key = norm_key(a)
        if key in colmap:
            return colmap[key]
    return None


def get_val(row, colmap, aliases, default=''):
    col = find_col(colmap, aliases if isinstance(aliases, (list, tuple)) else [aliases])
    if not col:
        return default
    val = row.get(col, '')
    if val is None:
        return default
    sval = str(val).strip()
    return sval if sval != '' and sval.lower() != 'nan' else default


def to_percent(val, default=0.0):
    try:
        f = float(val)
        if math.isnan(f):
            return default
    except Exception:
        return default
    if f <= 1.0:
        return max(0.0, min(100.0, f * 100.0))
    return max(0.0, min(100.0, f))


def booth_mask(series, booth_number):
    s = series.astype(str).str.strip()
    target = str(booth_number).strip()
    eq_str = s == target

    def to_intish(x):
        try:
            m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(x))
            if not m:
                return None
            return int(float(m.group(0)))
        except Exception:
            return None

    ints = s.map(to_intish)
    eq_int = ints == int(booth_number)
    return eq_str | eq_int
