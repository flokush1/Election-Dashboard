import pandas as pd
import pytest

from backend.app_factory import create_app
from backend.domain.assembly_slug import to_slug
from backend.domain.column_aliases import booth_mask, build_colmap, find_col
from backend.domain.voter_normalize import normalize_voter_payload
from backend.state.memory_store import store


@pytest.fixture()
def client(tmp_path):
    store.set_uploaded_voters([], [])
    app = create_app({'TESTING': True})
    return app.test_client()


def test_health_contract(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'healthy'
    assert 'model_loaded' in payload
    assert 'data_status' in payload
    assert 'csv_files_available' in payload['data_status']


def test_normalize_voter_payload_maps_codes():
    mapped = normalize_voter_payload({
        'voters id': 'ABC1234567',
        'name': 'Test Voter',
        'age': '42',
        'religion': 'Hindu',
        'economic_category_code': 'H',
        'Locality': 'MADIPUR'
    })
    assert mapped['voter_id'] == 'ABC1234567'
    assert mapped['economic_category'] == 'LOW INCOME AREAS'
    assert mapped['income'] == 'income_low'
    assert mapped['locality'] == 'MADIPUR'


def test_slug_and_column_aliases():
    assert to_slug('R.K. Puram') in {'r_k_puram', 'rk_puram'}
    df = pd.DataFrame({'Part No': ['017', '18'], 'Name': ['A', 'B']})
    colmap = build_colmap(df)
    assert find_col(colmap, ['partno', 'Part No']) == 'Part No'
    assert booth_mask(df['Part No'], 17).tolist()[0] is True


def test_predict_requires_model(client):
    response = client.post('/api/predict', json={'name': 'Nobody'})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_upload_and_search_contract(client):
    store.set_uploaded_voters(
        [{'voters id': 'XYZ999'}],
        [{'voter_id': 'XYZ999', 'name': 'Sample'}]
    )
    missing = client.post('/api/search-voter', json={'voter_id': 'NOPE'})
    assert missing.get_json()['found'] is False
    found = client.post('/api/search-voter', json={'voter_id': 'xyz999'})
    payload = found.get_json()
    assert payload['success'] is True
    assert payload['voter']['voter_id'] == 'XYZ999'


def test_preview_requires_assembly(client):
    response = client.get('/api/assembly-data-preview')
    assert response.status_code == 400
    assert 'assembly' in response.get_json()['error']
