from pathlib import Path

PUBLIC_ROOT = Path(__file__).resolve().parents[1] / 'public' / 'data'
FORBIDDEN = {'voter_id', 'voters id', 'full address', 'house number', 'epic'}


def test_public_data_has_no_voter_pii_columns():
    if not PUBLIC_ROOT.exists():
        return
    for path in PUBLIC_ROOT.rglob('*'):
        if path.suffix.lower() not in {'.json', '.geojson', '.csv'}:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore').lower()
        for token in FORBIDDEN:
            # Allow sample-voters.json as a tiny non-production demo file.
            if path.name == 'sample-voters.json':
                continue
            if path.suffix.lower() == '.csv':
                header = text.splitlines()[0] if text else ''
                assert token not in header, f'{path} contains PII column {token}'
