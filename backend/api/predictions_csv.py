from flask import Blueprint, jsonify

from backend.services.booth_stats_service import csv_booth_statistics, individual_voter_prediction, voter_predictions_for_booth

bp = Blueprint('predictions_csv', __name__)


@bp.route('/api/voter-predictions/<assembly_name>/<int:booth_number>', methods=['GET'])
def voter_predictions_route(assembly_name, booth_number):
    payload, error = voter_predictions_for_booth(assembly_name, booth_number)
    if error:
        status = 404 if 'not found' in str(error.get('error', '')).lower() or 'No predictions' in str(error.get('error', '')) else 400
        return jsonify(error), status
    return jsonify(payload)


@bp.route('/api/booth-statistics/<assembly_name>/<int:booth_number>', methods=['GET'])
def booth_statistics_route(assembly_name, booth_number):
    payload, error = csv_booth_statistics(assembly_name, booth_number)
    if error:
        return jsonify(error), 404
    return jsonify(payload)


@bp.route('/api/voter-prediction/<voter_id>', methods=['GET'])
def individual_voter_route(voter_id):
    payload, error = individual_voter_prediction(voter_id)
    if error:
        return jsonify(error), 404
    return jsonify(payload)
