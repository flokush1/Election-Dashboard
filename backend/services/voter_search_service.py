import re

from backend.state.memory_store import store


def search_uploaded_voter(voter_id):
    target = str(voter_id or '').strip()
    if not target:
        return None
    if not store.mapped_data:
        return None
    for voter in store.mapped_data:
        voter_id_value = str(voter.get('voter_id', '')).strip()
        if voter_id_value.upper() == target.upper():
            return voter
    for voter in store.mapped_data:
        voter_id_value = str(voter.get('voter_id', '')).strip()
        if target.upper() in voter_id_value.upper() or voter_id_value.upper() in target.upper():
            return voter
    normalized_search = re.sub(r'[^A-Z0-9]', '', target.upper())
    for voter in store.mapped_data:
        voter_id_value = str(voter.get('voter_id', '')).strip()
        if re.sub(r'[^A-Z0-9]', '', voter_id_value.upper()) == normalized_search:
            return voter
    return None


def available_voters(limit=1000):
    return store.mapped_data[:limit]
