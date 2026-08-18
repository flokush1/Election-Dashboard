from functools import lru_cache
from threading import Lock

import pandas as pd

from backend.domain.assembly_slug import iter_prediction_csv_paths, resolve_predictions_file


_lock = Lock()


@lru_cache(maxsize=8)
def _read_csv_cached(path, modified_time):
    del modified_time
    return pd.read_csv(path, dtype=str, na_filter=False)


def read_predictions_csv(path):
    with _lock:
        import os
        return _read_csv_cached(path, os.path.getmtime(path)).copy()


def read_assembly_predictions(assembly_name):
    path = resolve_predictions_file(assembly_name)
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return read_predictions_csv(path), path


def search_voter_across_files(voter_id):
    import os
    from backend.domain.column_aliases import build_colmap, find_col

    for path in iter_prediction_csv_paths():
        if not os.path.exists(path):
            continue
        df = read_predictions_csv(path)
        colmap = build_colmap(df)
        voter_col = find_col(colmap, ['Voter_ID', 'Voter ID', 'voter_id', 'voters id', 'epic', 'epic_no'])
        if not voter_col:
            continue
        matches = df[df[voter_col].astype(str).str.strip() == str(voter_id).strip()]
        if not matches.empty:
            return matches, path
    return None, None
