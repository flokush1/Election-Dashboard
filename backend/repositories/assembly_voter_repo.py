from pathlib import Path

from backend.config import config


def resolve_voter_roll_path(filename):
    safe_name = Path(filename).name
    candidates = [
        config.VOTER_ROLLS_DIR / safe_name,
        config.PROJECT_ROOT / 'VoterID_Data_Assembly' / safe_name,
        config.PRIVATE_DATA_DIR / 'voter_rolls' / safe_name,
    ]
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        allowed_roots = [
            config.VOTER_ROLLS_DIR.resolve() if config.VOTER_ROLLS_DIR.exists() else config.VOTER_ROLLS_DIR,
            (config.PROJECT_ROOT / 'VoterID_Data_Assembly'),
        ]
        if any(str(resolved).startswith(str(root)) for root in allowed_roots if root):
            if resolved.exists():
                return resolved
    return None
