from fastapi.testclient import TestClient
from ml_project.infrastructure.serving import app


def test_predict_endpoint():
    client = TestClient(app)
    payload = {
        "features": {
            "a": 1,
            "b": "x",
            "income": 100,
            "expenses": 50,
            "date": "2021-01-01"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
