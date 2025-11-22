# Health Monitor ML

Small project to train a health status classifier and serve it with FastAPI.

Structure

- `app/` - FastAPI application
- `ml/` - training scripts and models

Quick start

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train (assumes `Health data.csv` is in the repo root):

```powershell
python ml\train_model.py
```

3. Run API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API

- `POST /predict` : JSON `{ "pulse": 75, "body_temperature": 36.5, "SpO2": 98 }`
- `GET /health` : health check
- `GET /model_info` : model metadata
