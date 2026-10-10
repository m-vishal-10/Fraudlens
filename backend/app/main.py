from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ml.mlflow_utils import list_runs
from ml.validation import FEATURE_COLUMNS, DatasetValidationError, validate_inference_frame
from backend.app.config import ARTIFACT_DIR, CORS_ORIGINS, MAX_UPLOAD_BYTES
from backend.app.schemas import (
    AlertResponse,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.model_service import ModelService

app = FastAPI(title="FraudLens API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
service = ModelService(ARTIFACT_DIR)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_available=service.available or (ARTIFACT_DIR / "model.joblib").exists())


@app.get("/api/v1/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    try:
        service.ensure_loaded()
    except Exception:
        return ModelInfo(available=False)
    return ModelInfo(
        available=True,
        selected_model=service.metadata.get("selected_model"),
        metadata={
            "random_state": service.metadata.get("random_state"),
            "split": service.metadata.get("split"),
            "test_metrics": service.metadata.get("test_metrics"),
        },
    )


@app.get("/api/v1/metrics")
def metrics() -> dict:
    try:
        service.ensure_loaded()
    except Exception:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable.")
    return {"evaluation_split": "held_out_test", **service.metadata["test_metrics"]}


def _read_artifact_json(filename: str, hint: str) -> dict:
    path = ARTIFACT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{filename} is unavailable. {hint}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"{filename} is unreadable: {exc}") from exc


@app.get("/api/v1/eda/summary")
def eda_summary() -> dict:
    return _read_artifact_json("eda_summary.json", "Run `python scripts/verify_dataset.py` first.")


@app.get("/api/v1/evaluation")
def evaluation() -> dict:
    report = _read_artifact_json("evaluation_report.json", "Run `python scripts/train.py` first.")
    metadata = _read_artifact_json("metadata.json", "Run `python scripts/train.py` first.")
    return {
        **report,
        "validation_metrics": metadata.get("validation_metrics"),
        "class_counts": metadata.get("class_counts"),
        "dataset_sha256": metadata.get("dataset_fingerprint"),
    }


# Only these generated files can be downloaded; request names are never joined into paths.
PUBLIC_ARTIFACTS = {
    **{name: Path("plots") / name for name in (
        "pr_curve.png",
        "confusion_matrix.png",
        "eda_class_balance.png",
        "eda_amount_by_class.png",
        "eda_time_by_class.png",
        "eda_key_features.png",
        "eda_class_correlations.png",
        "eda_feature_correlation.png",
    )},
    "top_200_test_queue.csv": Path("top_200_test_queue.csv"),
}


@app.get("/api/v1/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    relative = PUBLIC_ARTIFACTS.get(name)
    path = ARTIFACT_DIR / relative if relative else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found. Run the training and EDA scripts.")
    return FileResponse(path)


@app.get("/api/v1/experiments")
def experiments() -> dict:
    try:
        runs = list_runs()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"experiment": "FraudLens", "runs": runs}


@app.get("/api/v1/alerts/top-200", response_model=AlertResponse)
def top_200() -> AlertResponse:
    raise HTTPException(
        status_code=400,
        detail="No unlabeled operational batch is loaded. Upload a CSV to score a batch.",
    )


def _frame_to_alert_response(frame: pd.DataFrame) -> AlertResponse:
    records = frame.rename(columns={"Amount": "amount", "Time": "time"}).to_dict("records")
    return AlertResponse(alerts=records, queue_size=len(records), capacity=200)


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if set(request.features) != set(FEATURE_COLUMNS):
        raise HTTPException(status_code=422, detail=f"Exactly these features are required: {FEATURE_COLUMNS}")
    frame = pd.DataFrame([request.features], columns=FEATURE_COLUMNS)
    try:
        scored = service.score(frame)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(
        risk_score=float(scored.iloc[0]["risk_score"]),
        message="Prioritize this transaction for human review; the score does not confirm fraud.",
        reasons=scored.iloc[0]["reasons"],
    )


@app.post("/api/v1/alerts/upload", response_model=AlertResponse)
async def upload_alerts(file: UploadFile = File(...)) -> AlertResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the configured upload limit.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
        validate_inference_frame(frame)
        return _frame_to_alert_response(service.score(frame))
    except (pd.errors.ParserError, UnicodeDecodeError, DatasetValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
