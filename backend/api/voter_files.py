import pandas as pd
from flask import Blueprint, jsonify, request

from backend.domain.column_aliases import booth_mask, build_colmap, find_col
from backend.repositories.assembly_voter_repo import resolve_voter_roll_path

bp = Blueprint('voter_files', __name__)


@bp.route('/api/voter-preview', methods=['GET'])
def voter_preview():
    file_name = request.args.get('file')
    part_no = request.args.get('partNo', type=int)
    if not file_name:
        return jsonify({'error': 'file query parameter required'}), 400
    path = resolve_voter_roll_path(file_name)
    if path is None:
        return jsonify({'error': 'Voter roll file not found', 'file': file_name}), 404
    df = pd.read_excel(path)
    colmap = build_colmap(df)
    booth_col = find_col(colmap, ['partno', 'PartNo', 'part_no', 'Part No', 'booth', 'booth_no'])
    filtered_df = df
    if part_no is not None and booth_col:
        filtered_df = df[booth_mask(df[booth_col], part_no)]
    columns = list(filtered_df.columns)
    preview_data = filtered_df.head(50).fillna('').to_dict(orient='records')
    return jsonify({
        'columns': columns,
        'preview': preview_data,
        'totalRows': len(filtered_df),
        'partNo': part_no,
        'fileName': path.name,
        'message': f'Found {len(filtered_df)} voters for Part No {part_no}'
    })
