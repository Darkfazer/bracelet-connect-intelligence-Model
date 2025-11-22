from pathlib import Path
import sys
from fastapi.testclient import TestClient

# ensure package root on sys.path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from health_monitor_ml.app.main import app


def run_tests():
    client = TestClient(app)

    print("GET /health ->", client.get("/health").json())

    try:
        # reload artifacts so model_info reflects latest metadata
        try:
            reload_r = client.post("/reload")
            print("POST /reload ->", reload_r.status_code, reload_r.json())
        except Exception as e:
            print("POST /reload failed:", e)

        info = client.get("/model_info")
        print("GET /model_info ->", info.status_code, info.json())
    except Exception as e:
        print("GET /model_info failed:", e)

    payload = {"pulse": 75, "body_temperature": 36.5, "SpO2": 98}
    pred = client.post("/predict", json=payload)
    print("POST /predict ->", pred.status_code, pred.json())


if __name__ == '__main__':
    run_tests()
