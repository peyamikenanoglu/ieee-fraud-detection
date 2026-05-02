from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single transaction prediction request.

    The transaction dictionary can contain any raw IEEE-CIS transaction and identity
    fields. Missing fields are allowed; the preprocessing layer will impute or
    encode them using training artifacts.
    """

    transaction: Dict[str, Any] = Field(..., description="Raw transaction fields as key-value pairs")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional decision threshold")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    transactions: List[Dict[str, Any]] = Field(..., min_length=1, description="List of raw transaction records")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional decision threshold")


class PredictionResponse(BaseModel):
    model_name: str
    threshold: float
    prediction: int
    fraud_probability: float
    risk_label: str
    model_probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    model_name: str
    threshold: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    n_features: int
