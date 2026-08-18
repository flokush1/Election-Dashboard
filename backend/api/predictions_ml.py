from flask import Blueprint, jsonify, request

from backend.config import config
from backend.services.family_prediction_service import predict_family
from backend.services.model_service import get_predictor
from backend.services.prediction_service import predict_many, predict_one

bp = Blueprint('predictions_ml', __name__)


@bp.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    model_file.seek(0, 2)
    size_bytes = model_file.tell()
    model_file.seek(0)
    max_bytes = config.MAX_MODEL_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return jsonify({'error': f'Model file too large ({size_bytes/1024/1024:.1f}MB). Limit {config.MAX_MODEL_UPLOAD_MB}MB'}), 400
    model_data = model_file.read()
    if not model_data:
        return jsonify({'error': 'Uploaded file is empty'}), 400
    predictor = get_predictor()
    success, message = predictor.load_model(model_data)
    predictor.model_file_path = model_file.filename
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'model_type': 'VoterPredictor',
            'feature_count': len(predictor.feature_names) if predictor.feature_names else 'Unknown',
            'file_size_mb': f"{size_bytes/1024/1024:.2f}"
        })
    return jsonify({'error': message, 'file_size_mb': f"{size_bytes/1024/1024:.2f}"}), 500


@bp.route('/api/predict', methods=['POST'])
def predict_voter():
    result, error, mapped = predict_one(request.json)
    if error:
        status = 400 if 'not loaded' in error.lower() else 500
        return jsonify({'error': error}), status
    return jsonify({'success': True, 'prediction': result, 'mapped_voter': mapped})


@bp.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    payload = request.json or {}
    voters = payload.get('voters') or payload.get('data') or []
    results, errors = predict_many(voters)
    return jsonify({'success': True, 'results': results, 'errors': errors, 'total': len(results)})


@bp.route('/api/predict-family', methods=['POST'])
def predict_family_route():
    payload, error = predict_family(request.json or {})
    if error:
        return jsonify({'error': error}), 400
    return jsonify(payload)


@bp.route('/api/debug-voter', methods=['POST'])
def debug_voter():
    if not config.EXPOSE_DEBUG_ENDPOINTS:
        return jsonify({'error': 'Debug endpoints disabled'}), 404
    predictor = get_predictor()
    if not predictor.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    mapped_result, error, mapped = predict_one(request.json)
    return jsonify({
        'success': error is None,
        'mapped_voter': mapped,
        'prediction': mapped_result,
        'error': error,
        'model_status': predictor.status()
    })
