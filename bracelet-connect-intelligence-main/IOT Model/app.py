from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any


def load_model(model_path: Path):
    model_meta = joblib.load(model_path)
    if isinstance(model_meta, dict) and 'model' in model_meta:
        return model_meta['model'], model_meta.get('features')
    return model_meta, None


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Heart Rate (BPM)' in df.columns:
        df['Heart Rate (BPM)'] = pd.to_numeric(df['Heart Rate (BPM)'], errors='coerce')
    # rolling features
    df['hr_roll_mean_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).mean()
    df['hr_roll_std_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).std().fillna(0)
    df['hr_diff'] = df['Heart Rate (BPM)'].diff().fillna(0)
    # extract hour/minute from common timestamp columns
    ts_col = None
    for c in ['Timestamp', 'timestamp', 'time', 'Time']:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is not None:
        try:
            t = pd.to_datetime(df[ts_col], errors='coerce')
            df['hour'] = t.dt.hour.fillna(0).astype(int)
            df['minute'] = t.dt.minute.fillna(0).astype(int)
        except Exception:
            pass
    return df


# Attempt to import FastAPI; keep file runnable even if FastAPI isn't installed
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    fastapi_available = True
except Exception:
    fastapi_available = False


def predict_from_records(model, features: List[str], records: List[Dict[str, Any]]) -> List[float]:
    df = pd.DataFrame(records)
    df = make_features(df)
    # align features
    if features is None:
        features = ['Heart Rate (BPM)', 'hr_roll_mean_5', 'hr_roll_std_5', 'hr_diff']
    existing = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        # warn, but proceed with existing features if they match model
        pass
    if not existing:
        raise ValueError('No input features found in payload; available columns: ' + ','.join(df.columns.tolist()))
    X = df[existing].values
    preds = model.predict(X)
    return [float(p) for p in preds]


if fastapi_available:
    app = FastAPI(title='IOT Model Inference')

    class PredictRequest(BaseModel):
        records: List[Dict[str, Any]]

    @app.get('/')
    def health():
        return {'status': 'ok'}

    @app.post('/predict')
    def predict(req: PredictRequest):
        model_path = Path(__file__).resolve().parent / 'model_spo2_rf.joblib'
        if not model_path.exists():
            raise HTTPException(status_code=500, detail=f'model not found at {model_path}')
        model, features = load_model(model_path)
        try:
            preds = predict_from_records(model, features, req.records)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {'predictions': preds}


def _self_test():
    """Run a quick self-test using the first rows of the synthetic CSV and print results.
    This function avoids requiring FastAPI/uvicorn to be installed.
    """
    base = Path(__file__).resolve().parent
    csv_candidates = [base / 'synthetic_spo2_hr_1000.csv', base.parent / 'synthetic_spo2_hr_1000.csv']
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        print('No dataset found for self-test. Expected synthetic_spo2_hr_1000.csv in project root or IOT Model folder.')
        return
    model_path = base / 'model_spo2_rf.joblib'
    if not model_path.exists():
        print(f'Model artifact not found at {model_path}. Run training first.')
        return
    print('Loading model...')
    model, features = load_model(model_path)
    df = pd.read_csv(csv_path)
    df = make_features(df)
    # prepare sample input
    df = df.dropna(subset=['Heart Rate (BPM)'])
    sample = df.head(5)
    records = sample.to_dict(orient='records')
    print('Predicting on a small sample...')
    preds = predict_from_records(model, features, records)
    for i, p in enumerate(preds):
        print(f'row {i}: pred_SpO2 = {p:.3f}')


if __name__ == '__main__':
    # Run a self-test if module executed directly. If FastAPI and uvicorn are installed,
    # run the server with `uvicorn "IOT Model.app:app" --reload --port 8000` instead.
    _self_test()
