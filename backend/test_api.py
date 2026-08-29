from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_project_explainability_fields_are_present():
    response = client.get('/api/projects?page=1&limit=1')
    assert response.status_code == 200
    payload = response.json()
    assert payload['items']

    item = payload['items'][0]
    assert 'confidence' in item
    assert 'score_breakdown' in item
    assert 'provenance' in item
    assert 'used_fields' in item['provenance']
    assert 'matching_method' in item['provenance']
