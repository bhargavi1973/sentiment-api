import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

MOCK_RESULT = {
    "text": "test",
    "label": "POSITIVE",
    "confidence": 0.95,
    "scores": {"POSITIVE": 0.95, "NEGATIVE": 0.03, "NEUTRAL": 0.02}
}

@pytest.fixture
def client():
    with patch("app.model.predict", return_value=MOCK_RESULT):
        from app.main import app
        yield TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_returns_200(client):
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200

def test_predict_returns_label(client):
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.json()["label"] == "POSITIVE"

def test_empty_text_returns_400(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400

def test_long_text_returns_400(client):
    response = client.post("/predict", json={"text": "a" * 513})
    assert response.status_code == 400