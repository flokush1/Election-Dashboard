from backend.domain.voter_normalize import normalize_voter_payload
from backend.services.model_service import get_predictor


def predict_one(voter_input):
    predictor = get_predictor()
    mapped = normalize_voter_payload(voter_input or {}, row_index_fallback=1)
    if not predictor.model_loaded:
        return None, 'Model not loaded. Please upload a model first.', mapped
    result, error = predictor.predict_voter(mapped)
    return result, error, mapped


def predict_many(voter_inputs, limit=None):
    rows = voter_inputs or []
    if limit:
        rows = rows[:limit]
    results = []
    errors = []
    for index, voter in enumerate(rows):
        result, error, _mapped = predict_one(voter)
        if error:
            errors.append({'index': index, 'error': error})
        else:
            results.append(result)
    return results, errors
