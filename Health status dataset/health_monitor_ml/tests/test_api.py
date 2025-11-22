from pathlib import Path
import sys
import pytest

# ensure repo root on sys.path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from fastapi.testclient import TestClient
from health_monitor_ml.app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_model_info_available():
    r = client.get("/model_info")
    assert r.status_code in (200, 503)


def test_predict_sample():
    payload = {"pulse": 75, "body_temperature": 36.5, "SpO2": 98}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "status" in j and "confidence" in j and "message" in j
