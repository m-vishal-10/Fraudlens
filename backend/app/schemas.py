from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ml.validation import FEATURE_COLUMNS


class HealthResponse(BaseModel):
    status: str
    model_available: bool


class PredictionRequest(BaseModel):
    features: dict[str, float]


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str
    description: str


class PredictionResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    message: str
    reasons: list[Reason] = []


class Alert(BaseModel):
    rank: int
    transaction_id: str
    risk_score: float
    amount: float
    time: float
    risk_category: str
    analyst_note: str
    reasons: list[Reason] = []


class AlertResponse(BaseModel):
    alerts: list[Alert]
    queue_size: int
    capacity: int
    labeled_evaluation: bool = False
    metrics: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    available: bool
    selected_model: str | None = None
    feature_columns: list[str] = FEATURE_COLUMNS
    metadata: dict[str, Any] | None = None
