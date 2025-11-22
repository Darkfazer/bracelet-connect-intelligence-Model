from pydantic import BaseModel


class HealthData(BaseModel):
    pulse: float
    body_temperature: float
    SpO2: float


class PredictionResponse(BaseModel):
    status: int
    confidence: float
    message: str


class ModelInfo(BaseModel):
    model_name: str
    best_test_f1: float
    details: dict
