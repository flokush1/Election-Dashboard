import pandas as pd

from backend.domain.voter_normalize import normalize_voter_payload
from backend.state.memory_store import store


def process_voter_excel(file_storage, sheet_name=None):
    requested_sheet = sheet_name
    if requested_sheet:
        df = pd.read_excel(file_storage, engine='openpyxl', sheet_name=requested_sheet)
        sheets_meta = [{'name': requested_sheet, 'rows': len(df)}]
    else:
        all_sheets = pd.read_excel(file_storage, engine='openpyxl', sheet_name=None)
        df_list = list(all_sheets.values())
        sheets_meta = [{'name': name, 'rows': len(sdf)} for name, sdf in all_sheets.items()]
        if not df_list:
            raise ValueError('No sheets found in Excel file')
        ordered_cols = list(df_list[0].columns)
        for sdf in df_list[1:]:
            for c in sdf.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)
        df = pd.concat([sdf.reindex(columns=ordered_cols) for sdf in df_list], ignore_index=True, sort=False)

    original_column_order = list(df.columns)
    raw_data = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in original_column_order:
            val = row[col]
            row_dict[col] = '' if pd.isna(val) or val == '' else val
        raw_data.append(row_dict)

    mapped_data = [normalize_voter_payload(row, row_index_fallback=i + 1) for i, row in enumerate(raw_data)]
    store.set_uploaded_voters(raw_data, mapped_data)
    return {
        'success': True,
        'raw_data': raw_data,
        'mapped_data': mapped_data,
        'total_rows': len(df),
        'columns': original_column_order,
        'rows_returned': len(df),
        'sheets': sheets_meta,
        'backend_cache_size': len(store.mapped_data)
    }
