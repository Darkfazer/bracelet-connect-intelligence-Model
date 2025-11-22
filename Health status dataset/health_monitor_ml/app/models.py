from pathlib import Path
import joblib
import numpy as np
import logging
from typing import Tuple, Any

logger = logging.getLogger("health_app.models")


def load_artifacts() -> Tuple[Any, Any, dict]:
    base = Path(__file__).resolve().parents[1]
    models_dir = base / "ml" / "models"
    model_path = models_dir / "health_classifier.pkl"
    scaler_path = models_dir / "scaler.pkl"
    metadata_path = models_dir / "model_metadata.json"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler not found. Run training script first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = {}
    if metadata_path.exists():
        import json

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, scaler, metadata


def predict(model, scaler, features: np.ndarray) -> Tuple[int, float]:
    """Return predicted class and confidence (max probability)."""
    X = np.array(features).reshape(1, -1)
    Xs = scaler.transform(X)

    # Some estimators may not have predict_proba; handle gracefully
    try:
        probs = model.predict_proba(Xs)
        confidence = float(np.max(probs))
    except Exception:
        # Fall back to decision_function or deterministic
        try:
            dec = model.decision_function(Xs)
            # convert decision to pseudo-prob via softmax
            exp = np.exp(dec - np.max(dec))
            probs = exp / np.sum(exp)
            confidence = float(np.max(probs))
        except Exception:
            confidence = 1.0

    pred = int(model.predict(Xs)[0])
    logger.info(f"Prediction: {pred}, confidence={confidence}")
    return pred, confidence
