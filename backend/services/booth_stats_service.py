from backend.domain.column_aliases import booth_mask, build_colmap, find_col, get_val, to_percent
from backend.domain.assembly_slug import filter_df_by_assembly
from backend.repositories.excel_repo import read_parliamentary_excel
from backend.repositories.predictions_csv_repo import read_assembly_predictions, search_voter_across_files


def _row_to_voter(row, colmap):
    return {
        'Voter_ID': get_val(row, colmap, ['voters id', 'Voter_ID', 'Voter ID', 'voter_id', 'VoterID', 'epic', 'epic_no', 'epic no'], ''),
        'section_no_road_name': get_val(row, colmap, ['section no & road name', 'section_no_road_name', 'section no and road name'], ''),
        'assembly_name': get_val(row, colmap, ['assembly name', 'assembly_name', 'ac_name', 'assembly'], ''),
        'name': get_val(row, colmap, ['name', 'voter_name'], ''),
        'relation_name': get_val(row, colmap, ['relation name', 'relation_name', 'father name', 'father_name', 'husband name', 'husband_name'], ''),
        'house_number': get_val(row, colmap, ['house number', 'house_number', 'houseno', 'house no'], ''),
        'Age': get_val(row, colmap, ['Age', 'age'], ''),
        'gender': get_val(row, colmap, ['gender', 'Gender', 'sex'], ''),
        'relation_type': get_val(row, colmap, ['relation type', 'relation_type'], ''),
        'having_deleted_tag': get_val(row, colmap, ['having deleted tag', 'having_deleted_tag'], ''),
        'address': get_val(row, colmap, ['full address', 'full_address', 'address'], '') or get_val(row, colmap, ['section no & road name', 'section_no_road_name'], ''),
        'Religion': get_val(row, colmap, ['Religion', 'religion'], ''),
        'Caste': get_val(row, colmap, ['Caste', 'caste', 'Category', 'category'], ''),
        'Locality': get_val(row, colmap, ['Locality', 'locality', 'area', 'location'], ''),
        'Economic': get_val(row, colmap, ['Economic', 'economic', 'economic_category', 'Economic Category'], ''),
        'Booth_ID': get_val(row, colmap, ['Booth_ID', 'Booth ID', 'booth_id', 'booth', 'booth_no', 'Booth_No', 'partno', 'part no'], ''),
        'predictions': {
            'turnout_prob': to_percent(get_val(row, colmap, ['turnout_prob', 'turnout', 'turnout_probability'], 0)),
            'prob_BJP': to_percent(get_val(row, colmap, ['prob_BJP', 'prob_bjp', 'bjp_prob', 'bjp'], 0)),
            'prob_Congress': to_percent(get_val(row, colmap, ['prob_Congress', 'prob_congress', 'congress_prob', 'congress'], 0)),
            'prob_AAP': to_percent(get_val(row, colmap, ['prob_AAP', 'prob_aap', 'aap_prob', 'aap'], 0)),
            'prob_Others': to_percent(get_val(row, colmap, ['prob_Others', 'prob_others', 'others_prob', 'others'], 0)),
            'prob_NOTA': to_percent(get_val(row, colmap, ['prob_NOTA', 'prob_nota', 'nota_prob', 'nota'], 0)),
        }
    }


def booth_excel_stats(assembly_name, booth_number):
    df = read_parliamentary_excel()
    df = filter_df_by_assembly(df, assembly_name)
    colmap = build_colmap(df)
    booth_col = find_col(colmap, [
        'booth', 'booth_no', 'booth_number', 'partno', 'part_no',
        'Booth_ID', 'Booth ID', 'booth_id', 'BoothNumber', 'boothno',
        'PartNo', 'Part No', 'Part_No', 'PARTNO'
    ])
    q = df
    if booth_col:
        q = df[booth_mask(df[booth_col], booth_number)]
    if q.empty:
        return None, {
            'error': 'No matching booth rows found',
            'assembly': assembly_name,
            'booth_number': booth_number,
            'debug': {
                'total_rows_after_assembly_filter': len(df),
                'booth_column_used': booth_col
            }
        }

    row = q.iloc[0]

    def get_float(r, candidates, default=0.0):
        val = get_val(r, colmap, candidates, default='')
        if val == '':
            return default
        return to_percent(val, default)

    parties = {
        'BJP': get_float(row, ['BJP', 'bjp', 'BJP_share', 'BJP_ratio', 'BJP_votes']),
        'Congress': get_float(row, ['Congress', 'congress', 'INC', 'Congress_share', 'Congress_ratio', 'Congress_votes']),
        'AAP': get_float(row, ['AAP', 'aap', 'AAP_share', 'AAP_ratio', 'AAP_votes']),
        'Others': get_float(row, ['Others', 'others', 'Others_share', 'Others_ratio', 'Others_votes']),
        'NOTA': get_float(row, ['NOTA', 'nota', 'NOTA_share', 'NOTA_ratio', 'NOTA_votes'])
    }
    total_voters = get_float(row, ['Total_Voters', 'total_voters', 'Total'], 0)
    expected_turnout_pct = get_float(row, ['Turnout', 'Expected_Turnout', 'turnout_probability'], 0)
    return {
        'success': True,
        'assembly': assembly_name,
        'booth_number': booth_number,
        'total_voters': int(total_voters) if total_voters else None,
        'expected_turnout': int((expected_turnout_pct / 100.0) * total_voters) if total_voters else None,
        'party_probabilities': parties,
        'expected_votes': parties,
        'demographics': {
            'age_groups': {
                '18-25': get_float(row, ['Age_18_25', '18-25', 'age 18-25', 'Age18_25']),
                '26-35': get_float(row, ['Age_26_35', '26-35', 'age 26-35', 'Age26_35']),
                '36-45': get_float(row, ['Age_36_45', '36-45', 'age 36-45', 'Age36_45']),
                '46-60': get_float(row, ['Age_46_60', '46-60', 'age 46-60', 'Age46_60']),
                '60+': get_float(row, ['Age_60_plus', '60+', 'age 60+', 'Age60_plus'])
            },
            'genders': {
                'male': get_float(row, ['male', 'Male', 'gender_male']),
                'female': get_float(row, ['female', 'Female', 'gender_female'])
            },
            'religions': {
                'hindu': get_float(row, ['Religion_Hindu', 'Hindu']),
                'muslim': get_float(row, ['Religion_Muslim', 'Muslim']),
                'sikh': get_float(row, ['Religion_Sikh', 'Sikh']),
                'christian': get_float(row, ['Religion_Christian', 'Christian']),
                'buddhist': get_float(row, ['Religion_Buddhist', 'Buddhist']),
                'jain': get_float(row, ['Religion_Jain', 'Jain']),
                'other': get_float(row, ['Religion_Other', 'Other'])
            },
            'castes': {
                'sc': get_float(row, ['Caste_SC', 'SC']),
                'obc': get_float(row, ['Caste_OBC', 'OBC']),
                'brahmin': get_float(row, ['Caste_Brahmin', 'Brahmin']),
                'kshatriya': get_float(row, ['Caste_Kshatriya', 'Kshatriya']),
                'vaishya': get_float(row, ['Caste_Vaishya', 'Vaishya']),
                'st': get_float(row, ['Caste_ST', 'ST'])
            }
        }
    }, None


def voter_predictions_for_booth(assembly_name, booth_number):
    df, path = read_assembly_predictions(assembly_name)
    colmap = build_colmap(df)
    booth_col = find_col(colmap, [
        'partno', 'PartNo', 'Part No', 'part no', 'Part_No', 'part_no', 'PARTNO',
        'Booth_ID', 'Booth ID', 'booth_id', 'booth', 'booth_no', 'Booth_No',
        'BoothNumber', 'booth_number', 'boothno'
    ])
    if not booth_col:
        return None, {'error': 'Booth/PartNo column not found in CSV'}
    booth_data = df[booth_mask(df[booth_col], booth_number)]
    if booth_data.empty:
        return None, {
            'error': f'No predictions found for booth {booth_number}',
            'booth_column': booth_col,
            'sample_values': df[booth_col].astype(str).head(10).tolist()
        }
    voters = [_row_to_voter(row.to_dict(), colmap) for _, row in booth_data.iterrows()]
    return {
        'success': True,
        'booth_number': booth_number,
        'assembly_name': assembly_name,
        'total_voters': len(voters),
        'voters': voters,
        'fileName': path
    }, None


def csv_booth_statistics(assembly_name, booth_number):
    payload, error = voter_predictions_for_booth(assembly_name, booth_number)
    if error:
        return None, error
    voters = payload['voters']
    total_voters = len(voters)

    def avg(key):
        values = [float(v['predictions'].get(key) or 0) for v in voters]
        return sum(values) / total_voters if total_voters else 0

    avg_prob_bjp = avg('prob_BJP')
    avg_prob_congress = avg('prob_Congress')
    avg_prob_aap = avg('prob_AAP')
    avg_prob_others = avg('prob_Others')
    avg_prob_nota = avg('prob_NOTA')
    expected_votes = {
        'BJP': round(avg_prob_bjp / 100.0 * total_voters, 2),
        'Congress': round(avg_prob_congress / 100.0 * total_voters, 2),
        'AAP': round(avg_prob_aap / 100.0 * total_voters, 2),
        'Others': round(avg_prob_others / 100.0 * total_voters, 2),
        'NOTA': round(avg_prob_nota / 100.0 * total_voters, 2),
    }
    predicted_winner = max(expected_votes, key=expected_votes.get) if expected_votes else None
    return {
        'success': True,
        'assembly': assembly_name,
        'booth_number': booth_number,
        'total_voters': total_voters,
        'predicted_winner': predicted_winner,
        'party_probabilities': {
            'BJP': round(avg_prob_bjp, 2),
            'Congress': round(avg_prob_congress, 2),
            'AAP': round(avg_prob_aap, 2),
            'Others': round(avg_prob_others, 2),
            'NOTA': round(avg_prob_nota, 2)
        },
        'expected_votes': expected_votes
    }, None


def individual_voter_prediction(voter_id):
    matches, path = search_voter_across_files(voter_id)
    if matches is None or matches.empty:
        return None, {'error': f'No prediction found for voter {voter_id}'}
    colmap = build_colmap(matches)
    voter = _row_to_voter(matches.iloc[0].to_dict(), colmap)
    return {'success': True, 'voter': voter, 'fileName': path}, None
