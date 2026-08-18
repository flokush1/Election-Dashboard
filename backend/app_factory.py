import logging
import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.api.electoral_excel import bp as electoral_excel_bp
from backend.api.health import bp as health_bp
from backend.api.predictions_csv import bp as predictions_csv_bp
from backend.api.predictions_ml import bp as predictions_ml_bp
from backend.api.voter_files import bp as voter_files_bp
from backend.api.voters_upload import bp as voters_upload_bp
from backend.config import config
from backend.ml.predictor import VoterPredictor
from backend.state.memory_store import store


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(config)
    if test_config:
        app.config.update(test_config)

    CORS(app, origins=config.CORS_ORIGINS or ['*'])
    logging.basicConfig(level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO))

    if store.predictor is None:
        store.set_predictor(VoterPredictor())

    app.register_blueprint(health_bp)
    app.register_blueprint(predictions_ml_bp)
    app.register_blueprint(voters_upload_bp)
    app.register_blueprint(electoral_excel_bp)
    app.register_blueprint(predictions_csv_bp)
    app.register_blueprint(voter_files_bp)

    @app.errorhandler(500)
    def handle_500(error):
        if request.path.startswith('/api/'):
            payload = {'error': 'Internal server error'}
            if config.INCLUDE_TRACEBACKS:
                payload['exception'] = str(error)
                payload['trace_tail'] = traceback.format_exc().splitlines()[-5:]
            return jsonify(payload), 500
        return error

    return app
