from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from .schemas import HealthData, PredictionResponse, ModelInfo
from . import models as model_utils
import logging
from pathlib import Path
import time

app = FastAPI(title="Health Monitor API")

# configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("health_app")

# Load artifacts at startup
MODEL = SCALER = METADATA = None
try:
    MODEL, SCALER, METADATA = model_utils.load_artifacts()
    logger.info("Loaded model and scaler")
except Exception as e:
    MODEL = SCALER = METADATA = None
    logger.warning(f"Model artifacts not loaded: {e}")


# Simple in-memory rate limiter per IP
class SimpleRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.store = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        hits = self.store.get(key, [])
        # keep only recent
        hits = [t for t in hits if t > now - self.window]
        if len(hits) >= self.max_requests:
            self.store[key] = hits
            return False
        hits.append(now)
        self.store[key] = hits
        return True


rate_limiter = SimpleRateLimiter(max_requests=30, window_seconds=60)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model_info", response_model=ModelInfo)
async def model_info():
    # Read metadata file on-demand so updates are visible without restart
    try:
        base = Path(__file__).resolve().parents[1]
        metadata_path = base / "ml" / "models" / "model_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=503, detail="Model metadata not available")
        import json

        with open(metadata_path, "r", encoding="utf-8") as f:
            md = json.load(f)

        # Backwards-compatible metadata parsing: accept either the original
        # keys (`best_model`, `best_test_f1`, `all_results`) or the hardened
        # keys (`model`, `test_f1`, etc.). Build a consistent response.
        model_name = md.get("best_model") or md.get("model") or md.get("model_name") or md.get("name") or "unknown"
        best_test_f1 = md.get("best_test_f1") or md.get("test_f1") or md.get("best_f1") or md.get("test_f1") or 0.0

        # prefer a structured `all_results` dict, otherwise include the raw metadata
        details = md.get("all_results") or md.get("details") or md
        # ensure numeric
        try:
            best_test_f1 = float(best_test_f1)
        except Exception:
            best_test_f1 = 0.0

        return ModelInfo(model_name=model_name, best_test_f1=best_test_f1, details=details)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to read model metadata")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/reload")
async def reload_model():
    """Reload model and scaler artifacts from disk without restarting the server."""
    global MODEL, SCALER, METADATA
    try:
        MODEL, SCALER, METADATA = model_utils.load_artifacts()
        # Log reload with timestamp and client IP if available
        logger.info("Reloaded model and scaler via /reload endpoint")
        return {"status": "reloaded"}
    except Exception as e:
        logger.exception("Failed to reload artifacts")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/raw_metadata")
async def raw_metadata():
    """Return raw model metadata file contents for debugging."""
    try:
        base = Path(__file__).resolve().parents[1]
        metadata_path = base / "ml" / "models" / "model_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="metadata not found")
        import json

        with open(metadata_path, "r", encoding="utf-8") as f:
            md = json.load(f)
        return md
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to read raw metadata")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HealthData, request: Request = None):
    # rate-limit by client IP
    client_ip = request.client.host if request and request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    # Basic validation / bounds checking
    if not (20 <= data.pulse <= 220):
        raise HTTPException(status_code=400, detail="pulse out of expected range")
    if not (30.0 <= data.body_temperature <= 45.0):
        raise HTTPException(status_code=400, detail="body_temperature out of expected range")
    if not (50 <= data.SpO2 <= 100):
        raise HTTPException(status_code=400, detail="SpO2 out of expected range")

    features = [data.pulse, data.body_temperature, data.SpO2]
    try:
        pred, confidence = model_utils.predict(MODEL, SCALER, features)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    status_messages = {
        0: "Normal",
        1: "Anomalie légère - Surveillance recommandée",
        2: "Anomalie sévère - Consultation médicale urgente",
    }

    # log the prediction
    log_line = f"{time.asctime()} | {client_ip} | {features} | pred={pred} | conf={confidence:.4f}\n"
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "predictions.log", "a", encoding="utf-8") as f:
        f.write(log_line)

    return PredictionResponse(status=int(pred), confidence=float(confidence), message=status_messages.get(pred, "Unknown"))
