"""
Basic smoke tests. Requires models to already be trained (run the training
scripts first) since the TestClient triggers the FastAPI lifespan startup.

Run: pytest tests/test_api.py
"""
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_shape():
    payload = {
        "Time": 406.0, "V1": -2.3, "V2": 1.4, "V3": -3.1, "V4": 2.2,
        "V5": -1.1, "V6": -0.5, "V7": -2.9, "V8": 1.1, "V9": -0.8,
        "V10": -2.5, "V11": 2.0, "V12": -3.0, "V13": 0.4, "V14": -3.6,
        "V15": 0.2, "V16": -1.7, "V17": -1.9, "V18": -0.3, "V19": 0.1,
        "V20": 0.05, "V21": 0.3, "V22": 0.7, "V23": -0.02, "V24": 0.1,
        "V25": 0.2, "V26": -0.1, "V27": 0.05, "V28": 0.02, "Amount": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert "risk_level" in body
    assert "top_features" in body
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
