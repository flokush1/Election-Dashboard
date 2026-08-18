from flask import Blueprint, jsonify

from backend.config import config
from backend.services.model_service import get_predictor, health_payload

bp = Blueprint('health', __name__)


@bp.route('/api/health', methods=['GET'])
def health_check():
    return jsonify(health_payload())


@bp.route('/api/model-features', methods=['GET'])
def model_features_debug():
    if not config.EXPOSE_DEBUG_ENDPOINTS:
        return jsonify({'error': 'Debug endpoints disabled'}), 404
    predictor = get_predictor()
    if not predictor.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    vec_features = []
    if predictor.vectorizer:
        if hasattr(predictor.vectorizer, 'feature_names_'):
            vec_features = list(predictor.vectorizer.feature_names_)
        elif hasattr(predictor.vectorizer, 'get_feature_names_out'):
            vec_features = list(predictor.vectorizer.get_feature_names_out())
    locality_features = [f for f in vec_features if 'locality' in f.lower()]
    return jsonify({
        'total_features': len(predictor.feature_names or []),
        'vectorizer_features': len(vec_features),
        'feature_breakdown': {
            'locality_count': len(locality_features),
            'age_count': len([f for f in vec_features if 'age' in f.lower()]),
            'religion_count': len([f for f in vec_features if 'religion' in f.lower()]),
            'caste_count': len([f for f in vec_features if 'caste' in f.lower()]),
            'economic_count': len([f for f in vec_features if 'economic' in f.lower() or 'income' in f.lower()])
        },
        'locality_features': locality_features[:30],
        'age_features': [f for f in vec_features if 'age' in f.lower()],
        'religion_features': [f for f in vec_features if 'religion' in f.lower()],
        'caste_features': [f for f in vec_features if 'caste' in f.lower()],
        'economic_features': [f for f in vec_features if 'economic' in f.lower() or 'income' in f.lower()],
        'numeric_features': ['land_rate', 'construction_cost', 'population', 'male_female_ratio']
    })
