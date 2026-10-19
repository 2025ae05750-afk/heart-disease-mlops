import joblib
import pandas as pd
from fastapi.testclient import TestClient
from src import api
from src.config import FEATURES

class FakeModel:
    def predict(self, X):
        assert list(X.columns) == FEATURES
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]

def test_health(monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel())
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict(monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel())
    client = TestClient(api.app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["confidence"] == 0.8
