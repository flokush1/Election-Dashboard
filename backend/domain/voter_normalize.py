VALID_ECON_CATEGORIES = [
    'LOW INCOME AREAS',
    'LOWER MIDDLE CLASS',
    'MIDDLE CLASS',
    'UPPER MIDDLE CLASS',
    'PREMIUM AREAS',
]

ECON_CODE_MAP = {
    'H': 'LOW INCOME AREAS',
    'C': 'PREMIUM AREAS',
    '1': 'LOW INCOME AREAS',
    'L': 'LOW INCOME AREAS',
    '2': 'LOWER MIDDLE CLASS',
    'LM': 'LOWER MIDDLE CLASS',
    '3': 'MIDDLE CLASS',
    'M': 'MIDDLE CLASS',
    '4': 'UPPER MIDDLE CLASS',
    'UM': 'UPPER MIDDLE CLASS',
    '5': 'PREMIUM AREAS',
    'P': 'PREMIUM AREAS',
}


def find_value(source, possible_names, default=None):
    if not isinstance(possible_names, (list, tuple)):
        possible_names = [possible_names]
    if not isinstance(source, dict):
        return default

    lower_map = {str(k).lower().strip(): v for k, v in source.items()}
    for name in possible_names:
        key = str(name).lower().strip()
        if key in lower_map:
            v = lower_map[key]
            if v is None:
                continue
            s = str(v).strip()
            if s != "" and s.lower() not in {"nan", "na", "n/a", "none", "null"}:
                return s

    for name in possible_names:
        target = str(name).lower().strip()
        for k, v in source.items():
            k_norm = str(k).lower().strip()
            if target in k_norm:
                if v is None:
                    continue
                s = str(v).strip()
                if s != "" and s.lower() not in {"nan", "na", "n/a", "none", "null"}:
                    return s
    return default


def normalize_economic_category(raw_econ, econ_code=None):
    code = str(econ_code or '').strip().upper()
    if code in ECON_CODE_MAP:
        return ECON_CODE_MAP[code], code or None

    econ_full = str(raw_econ).strip().upper() if raw_econ else 'MIDDLE CLASS'
    if econ_full not in VALID_ECON_CATEGORIES:
        econ_full = 'MIDDLE CLASS'
    return econ_full, code or None


def income_from_economic(econ_full):
    if econ_full in ['PREMIUM AREAS', 'UPPER MIDDLE CLASS']:
        return 'income_high'
    if econ_full in ['LOW INCOME AREAS', 'LOWER MIDDLE CLASS']:
        return 'income_low'
    return 'income_middle'


def normalize_voter_payload(voter_input, row_index_fallback=1):
    from backend.utils.numbers import safe_int
    from backend.utils.numbers import to_float_safe as safe_float

    raw_econ = find_value(voter_input, ['economic_category', 'Economic Category', 'Economic'])
    econ_code = find_value(voter_input, ['economic_category_code', 'econ_code', 'eco_code'])
    econ_full, econ_code = normalize_economic_category(raw_econ, econ_code)
    income_level = income_from_economic(econ_full)

    voter_id = (
        find_value(voter_input, ['voter_id', 'voter id', 'voters id', 'VoterID', 'EPIC', 'epic no', 'epic number', 'id'])
        or voter_input.get('voter_id')
        or f'VOTER_{row_index_fallback:05d}'
    )

    raw_religion = find_value(voter_input, ['religion', 'Religion']) or voter_input.get('religion')
    religion_upper = str(raw_religion).strip().upper() if raw_religion else 'HINDU'
    raw_caste = find_value(voter_input, ['caste', 'Caste', 'Category', 'category', 'Social Category', 'social_category']) or voter_input.get('caste')
    caste_upper = str(raw_caste).strip().upper() if raw_caste else ''
    locality = (find_value(voter_input, ['Locality', 'locality', 'Area', 'area', 'Location', 'location']) or voter_input.get('locality') or voter_input.get('Locality') or '')

    mapped_voter = {
        'voter_id': voter_id,
        'name': find_value(voter_input, ['name', 'Name', 'voter_name', 'Voter Name', 'relation name']) or voter_input.get('name') or 'Unknown',
        'age': safe_int(find_value(voter_input, ['age', 'Age']), 30),
        'gender': (find_value(voter_input, ['gender', 'Gender', 'sex', 'Sex']) or voter_input.get('gender') or 'Unknown').upper(),
        'religion': religion_upper,
        'caste': caste_upper,
        'economic_category': econ_full,
        'economic_category_code': econ_code,
        'income': income_level,
        'Locality': locality,
        'locality': locality,
        'assembly': find_value(voter_input, ['assembly name', 'assembly', 'Assembly', 'Constituency', 'AC', 'assembly_constituency', 'ac_name']) or voter_input.get('assembly') or 'Unknown',
        'section_road': find_value(voter_input, ['section no & road name', 'section_road']) or voter_input.get('section_road') or 'Unknown',
        'full_address': find_value(voter_input, ['full_address', 'full address', 'Full Address', 'complete_address', 'Complete Address', 'residential_address', 'Residential Address', 'address', 'Address']) or voter_input.get('full_address') or voter_input.get('address') or 'Unknown',
        'partno': safe_int(find_value(voter_input, ['partno', 'PartNo', 'part_no', 'Part No', 'Booth_ID', 'booth_id', 'booth_no', 'Booth_No']), row_index_fallback),
        'booth_no': safe_int(find_value(voter_input, ['booth_no', 'Booth_No', 'Booth_ID', 'booth_id', 'partno', 'PartNo', 'part_no', 'Part No']), row_index_fallback),
        'land_rate_per_sqm': safe_float(find_value(voter_input, ['land_rate_per_sqm', 'land_rate']), 0.0),
        'construction_cost_per_sqm': safe_float(find_value(voter_input, ['construction_cost_per_sqm', 'construction_cost']), 0.0),
        'population': safe_float(find_value(voter_input, ['population', 'Population']), 0.0),
        'MaleToFemaleRatio': safe_float(find_value(voter_input, ['MaleToFemaleRatio', 'male_female_ratio', 'male_to_female_ratio']), 1.0),
        'male_female_ratio': safe_float(find_value(voter_input, ['male_female_ratio', 'MaleToFemaleRatio', 'male_to_female_ratio']), 1.0),
        'household_id': find_value(voter_input, ['household_id']) or voter_input.get('household_id'),
        'family_id_main': find_value(voter_input, ['family_id_main']) or voter_input.get('family_id_main'),
        'core_family_id': find_value(voter_input, ['core_family_id']) or voter_input.get('core_family_id'),
        'core_family_head': find_value(voter_input, ['core_family_head']) or voter_input.get('core_family_head'),
        'family_head': find_value(voter_input, ['family_head']) or voter_input.get('family_head'),
        'core_family_size': safe_int(find_value(voter_input, ['core_family_size']), 1),
        'main_family_size': safe_int(find_value(voter_input, ['main_family_size']), 1),
        'family_by_chain': find_value(voter_input, ['family_by_chain']) or voter_input.get('family_by_chain'),
        'family_by_chain_id': find_value(voter_input, ['family_by_chain_id']) or voter_input.get('family_by_chain_id'),
        'house_number': find_value(voter_input, ['house number', 'house_number', 'houseno', 'house no']) or voter_input.get('house_number') or 'Unknown',
        'relation_type': find_value(voter_input, ['relation type', 'relation_type']) or voter_input.get('relation_type') or 'Unknown',
        'having_deleted_tag': find_value(voter_input, ['having deleted tag', 'having_deleted_tag']) or voter_input.get('having_deleted_tag') or 'No',
        'houseno_base': find_value(voter_input, ['houseno_base']) or voter_input.get('houseno_base'),
        'houseno_normalized': find_value(voter_input, ['houseno_normalized']) or voter_input.get('houseno_normalized'),
        'addressbasekeynf': find_value(voter_input, ['addressbasekeynf']) or voter_input.get('addressbasekeynf'),
        'surname_effective': find_value(voter_input, ['surname_effective']) or voter_input.get('surname_effective'),
        'head_generation_level': safe_int(find_value(voter_input, ['head_generation_level']), 0),
    }
    if mapped_voter['booth_no'] == 0 and mapped_voter['partno'] != 0:
        mapped_voter['booth_no'] = mapped_voter['partno']
    return mapped_voter
