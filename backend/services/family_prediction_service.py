from backend.services.model_service import get_predictor
from backend.services.prediction_service import predict_one


def _member_payload(member, family_type, result):
    return {
        'name': member.get('name', 'Family Member'),
        'voter_id': member.get('voter_id', ''),
        'age': member.get('age', 0),
        'family_type': family_type,
        'predicted_party': result.get('predicted_party', 'Unknown'),
        'party_probabilities': result.get('party_probabilities', {}),
        'turnout_probability': result.get('turnout_probability', 0.0),
        'confidence_level': result.get('confidence_level', 'Medium'),
        'model_confidence': result.get('model_confidence', '0%')
    }


def predict_family(payload):
    predictor = get_predictor()
    if not predictor.model_loaded:
        return None, 'Model not loaded. Please upload a model first.'

    core = payload.get('coreMembers') or payload.get('core_family') or payload.get('family') or []
    chain = payload.get('chainMembers') or payload.get('chain_family') or payload.get('extended_family') or []
    results = []
    for voter in core[:5]:
        result, error, _mapped = predict_one(voter)
        if result:
            results.append(_member_payload(voter, 'core', result))
    for voter in chain[:5]:
        result, error, _mapped = predict_one(voter)
        if result:
            results.append(_member_payload(voter, 'chain', result))
    return {'success': True, 'family_predictions': results}, None
