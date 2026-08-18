from backend.domain.assembly_slug import list_prediction_files
from backend.ml.predictor import VoterPredictor
from backend.repositories.excel_repo import excel_available
from backend.state.memory_store import store


def get_predictor():
    if store.predictor is None:
        store.set_predictor(VoterPredictor())
    return store.predictor


def health_payload():
    predictor = get_predictor()
    csv_files = list_prediction_files()
    return {
        'status': 'healthy',
        'model_loaded': bool(predictor.model_loaded),
        'model_file': predictor.model_file_path,
        'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
        'party_count': len(predictor.party_names) if predictor.party_names else 0,
        'has_vectorizer': predictor.vectorizer is not None,
        'has_scaler': predictor.scaler is not None,
        'model_arrays_loaded': {
            'beta_P': predictor._beta_P_array is not None,
            'gamma0': predictor._gamma0_array is not None,
            'booth_effects_P': predictor._booth_effects_P_array is not None
        },
        'data_status': {
            'uploaded_voters_count': len(store.mapped_data),
            'has_raw_data': len(store.raw_data) > 0,
            'sample_voter_ids': [v.get('voter_id', 'NO_ID') for v in store.mapped_data[:5]] if store.mapped_data else [],
            'excel_available': excel_available(),
            'csv_files_available': csv_files,
            'csv_count': sum(1 for available in csv_files.values() if available)
        },
        'setup_notes': {
            'ready_to_use': any(csv_files.values()),
            'excel_note': 'Excel file available for advanced features' if excel_available() else 'Excel file not found - some endpoints will return 404 (dashboard still works)',
            'csv_note': f'{sum(1 for available in csv_files.values() if available)}/{len(csv_files)} prediction CSV files found',
            'model_note': 'ML model loaded' if predictor.model_loaded else 'No ML model loaded - upload via /api/upload-model or predictions from CSV will be used'
        }
    }
