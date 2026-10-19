import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .config import FEATURES, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="MLOps Assignment 01 prediction service."
)

model = None

class HeartDiseaseRequest(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(...)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=7)

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probability_absence: float
    probability_presence: float

@app.on_event("startup")
def load_model():
    global model
    if not Path(MODEL_PATH).exists():
        logger.error("Model artifact not found: %s", MODEL_PATH)
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        request.method, request.url.path, response.status_code, duration_ms
    )
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metadata")
def metadata():
    return {
        "features": FEATURES,
        "target": "0 = absence, 1 = presence",
        "model_artifact": str(MODEL_PATH),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: HeartDiseaseRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")

    payload = request.model_dump()
    row = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)

    try:
        pred = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        prediction=pred,
        confidence=float(max(probabilities)),
        probability_absence=float(probabilities[0]),
        probability_presence=float(probabilities[1]),
    )

Instrumentator().instrument(app).expose(app)
