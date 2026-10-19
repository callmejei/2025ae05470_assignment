"""FastAPI service exposing the heart-disease classifier."""
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest,
)
from pydantic import BaseModel, Field

from src.config import MODEL_FILE
from src.predict import load_model, predict

MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODEL_FILE))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": self.formatTime(record), "level": record.levelname,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart-api")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class",
                      ["label"])


class Patient(BaseModel):
    age: int = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="chest pain type 1-4")
    trestbps: int = Field(..., ge=50, le=250, description="resting BP mmHg")
    chol: int = Field(..., ge=50, le=700, description="serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="fasting sugar > 120")
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250, description="max heart rate")
    exang: int = Field(..., ge=0, le=1, description="exercise angina")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3, description="major vessels coloured")
    thal: int = Field(..., description="3 = normal, 6 = fixed, 7 = reversible")


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    model_version: str


app = FastAPI(title="Heart Disease Risk API", version=MODEL_VERSION)
model = None


@app.on_event("startup")
def _load():
    global model
    model = load_model(MODEL_PATH)
    log.info("model loaded", extra={"extra_fields": {"path": str(MODEL_PATH)}})


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, status).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method,
                "path": path, "status": status,
                "latency_ms": round(elapsed * 1000, 2)}})


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs",
            "endpoints": ["/health", "/predict", "/metrics"]}


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "loading",
            "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict_one(patient: Patient):
    if model is None:
        raise HTTPException(503, "model not loaded")
    result = predict(model, [patient.model_dump()])[0]
    PREDICTIONS.labels(result["label"]).inc()
    log.info("prediction", extra={"extra_fields": {
        "input": patient.model_dump(), **result}})
    return {**result, "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
