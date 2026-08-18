from flask import Blueprint, jsonify, request

from backend.domain.assembly_slug import filter_df_by_assembly
from backend.repositories.excel_repo import read_parliamentary_excel, resolve_excel_path
from backend.services.booth_stats_service import booth_excel_stats

bp = Blueprint('electoral_excel', __name__)


@bp.route('/api/booth-excel-stats/<assembly_name>/<int:booth_number>', methods=['GET'])
def booth_excel_stats_route(assembly_name, booth_number):
    try:
        payload, error = booth_excel_stats(assembly_name, booth_number)
        if error:
            return jsonify(error), 404
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({'error': 'Parliament Excel file not found', 'path': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/api/parliament-data-preview', methods=['GET'])
def parliament_data_preview():
    try:
        limit = request.args.get('limit', default=20, type=int)
        limit = max(1, min(limit, 300))
        df = read_parliamentary_excel(nrows=limit)
        columns_meta = [{'name': col, 'dtype': str(df[col].dtype)} for col in df.columns]
        return jsonify({
            'success': True,
            'file': resolve_excel_path().name,
            'row_count_preview': len(df),
            'columns': columns_meta,
            'rows': df.fillna('').to_dict(orient='records')
        })
    except FileNotFoundError as exc:
        return jsonify({'error': 'Parliament Excel file not found', 'path': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/api/assembly-data-preview', methods=['GET'])
def assembly_data_preview():
    try:
        assembly = request.args.get('assembly')
        if not assembly:
            return jsonify({'error': 'assembly query parameter required'}), 400
        limit = request.args.get('limit', default=20, type=int)
        limit = max(1, min(limit, 300))
        df_full = read_parliamentary_excel()
        filtered = filter_df_by_assembly(df_full, assembly)
        df = filtered.head(limit)
        columns_meta = [{'name': col, 'dtype': str(df[col].dtype)} for col in df.columns]
        return jsonify({
            'success': True,
            'file': resolve_excel_path().name,
            'assembly': assembly,
            'row_count_preview': len(df),
            'total_matching_rows': len(filtered),
            'columns': columns_meta,
            'rows': df.fillna('').to_dict(orient='records')
        })
    except FileNotFoundError as exc:
        return jsonify({'error': 'Parliament Excel file not found', 'path': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
