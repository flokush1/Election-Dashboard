from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from backend.config import config
from backend.services.upload_session_service import process_voter_excel
from backend.services.voter_search_service import available_voters, search_uploaded_voter

bp = Blueprint('voters_upload', __name__)


@bp.route('/api/upload-voter-data', methods=['POST'])
def upload_voter_data():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'success': False, 'error': 'Please upload an Excel file (.xlsx or .xls)'}), 400
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > config.MAX_EXCEL_UPLOAD_MB:
        return jsonify({
            'success': False,
            'error': f'File too large: {file_size_mb:.1f}MB. Server processing limited to {config.MAX_EXCEL_UPLOAD_MB}MB.'
        }), 400
    try:
        payload = process_voter_excel(file, request.form.get('sheet_name') or request.args.get('sheet_name'))
        payload['file_info'] = {'name': filename, 'size_mb': round(file_size_mb, 2)}
        return jsonify(payload)
    except Exception as exc:
        return jsonify({'success': False, 'error': f'Server error processing file: {exc}'}), 500


@bp.route('/api/search-voter', methods=['POST'])
def search_voter():
    payload = request.json or {}
    voter_id = str(payload.get('voter_id') or payload.get('voterId') or payload.get('id') or '').strip()
    if not voter_id:
        return jsonify({'error': 'No voter ID provided'}), 400
    from backend.state.memory_store import store
    if not store.mapped_data:
        return jsonify({'error': 'No voter data uploaded yet. Please upload voter data first.'}), 400
    voter = search_uploaded_voter(voter_id)
    if not voter:
        available_ids = [v.get('voter_id', '') for v in store.mapped_data[:10]]
        return jsonify({
            'success': False,
            'error': f'Voter ID "{voter_id}" not found in uploaded data',
            'found': False,
            'available_sample': available_ids,
            'total_voters_in_cache': len(store.mapped_data)
        })
    return jsonify({'success': True, 'voter': voter, 'found': True})


@bp.route('/api/available-voters', methods=['GET'])
def available_voters_route():
    return jsonify({'success': True, 'voters': available_voters(), 'total': len(available_voters())})
