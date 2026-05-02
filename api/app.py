from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from api.prediction_service import FraudPredictionService
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Production-style API for v7 final ensemble fraud prediction.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "message": "IEEE-CIS Fraud Detection API is running.",
        "docs_url": "http://127.0.0.1:8000/docs",
        "health_url": "http://127.0.0.1:8000/health",
        "predict_endpoint": "POST /predict"
    }

service = FraudPredictionService(project_root=PROJECT_ROOT)


@app.on_event("startup")
def load_model_artifacts() -> None:
    service.load()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_loaded else "not_loaded",
        model_loaded=service.is_loaded,
        model_name=service.model_name,
        n_features=service.n_features,
    )


@app.get("/model-info")
def model_info() -> dict:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    return {
        "model_name": service.model_name,
        "n_features": service.n_features,
        "config": service.config,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    try:
        result = service.predict([request.transaction], threshold=request.threshold)[0]
        return PredictionResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        results = service.predict(request.transactions, threshold=request.threshold)
        threshold = request.threshold
        if threshold is None:
            threshold = float(service.config.get("threshold_best_f1", 0.71))
        return BatchPredictionResponse(
            model_name=service.model_name,
            threshold=float(threshold),
            predictions=[PredictionResponse(**result) for result in results],
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
