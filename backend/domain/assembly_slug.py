import os
import re

import pandas as pd

from backend.config import config
from backend.domain.column_aliases import build_colmap, find_col


def to_slug(s: str) -> str:
    s = str(s or '').strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return s.strip("_")


def _candidate_paths(filename):
    roots = [
        os.getcwd(),
        str(config.PROJECT_ROOT),
        str(config.PREDICTIONS_DIR),
        str(config.PRIVATE_DATA_DIR / 'predictions'),
        str(config.DATA_ROOT),
    ]
    seen = set()
    for root in roots:
        path = os.path.join(root, filename)
        if path not in seen:
            seen.add(path)
            yield path
        if os.path.exists(filename) and filename not in seen:
            seen.add(filename)
            yield filename


def resolve_predictions_file(assembly_name: str) -> str:
    slug = to_slug(assembly_name)
    base_tokens = [t for t in re.split(r"[^0-9a-z]+", str(assembly_name).strip().lower()) if t]
    letter_split_tokens = [('_'.join(list(t)) if t.isalpha() and 1 < len(t) <= 3 else t) for t in base_tokens]
    slug_variant_letter_split = '_'.join(letter_split_tokens)

    slug_variants = [slug]
    if slug_variant_letter_split and slug_variant_letter_split != slug:
        slug_variants.append(slug_variant_letter_split)

    candidates = []
    for sv in slug_variants:
        candidates.append(f"predictions_{sv}.csv")
        candidates.append(f"predictions_{sv}")

    for path in candidates:
        for resolved in _candidate_paths(path):
            if os.path.exists(resolved):
                return resolved

    legacy_map = {
        'new_delhi': 'predictions_new_delhi.csv',
        'r_k_puram': 'predictions_r_k_puram.csv',
        'rk_puram': 'predictions_r_k_puram.csv',
    }
    fallback = legacy_map.get(slug)
    if fallback:
        for resolved in _candidate_paths(fallback):
            if os.path.exists(resolved):
                return resolved
    return os.path.join(str(config.PREDICTIONS_DIR), candidates[0])


def list_prediction_files():
    names = {
        'new_delhi': 'predictions_new_delhi.csv',
        'r_k_puram': 'predictions_r_k_puram.csv',
        'newdelhi_voter': 'newdelhi_voter_predictions.csv',
    }
    found = {}
    for key, name in names.items():
        found[key] = any(os.path.exists(path) for path in _candidate_paths(name))
    return found


def iter_prediction_csv_paths():
    seen = set()
    search_dirs = [
        config.PREDICTIONS_DIR,
        config.PROJECT_ROOT,
        config.DATA_ROOT,
    ]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.lower().endswith('.csv'):
                continue
            if 'prediction' not in name.lower() and not name.lower().startswith('predictions_'):
                continue
            path = os.path.join(directory, name)
            if path not in seen:
                seen.add(path)
                yield path


def filter_df_by_assembly(df: pd.DataFrame, assembly_name: str) -> pd.DataFrame:
    colmap = build_colmap(df)
    asm_col = find_col(colmap, ['assembly name', 'AssemblyName', 'assembly_name', 'assembly', 'ac_name', 'AC', 'constituency'])
    if not asm_col:
        return df

    def norm_asm(s: str) -> str:
        s = str(s or '').strip().lower()
        return re.sub(r"[^0-9a-z]+", "", s)

    try:
        target_norm = norm_asm(assembly_name)
        series_norm = df[asm_col].astype(str).map(norm_asm)
        mask = series_norm.str.contains(target_norm, regex=False, na=False)
        return df[mask]
    except Exception:
        return df
