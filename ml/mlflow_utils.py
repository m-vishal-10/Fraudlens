from __future__ import annotations

import os
from pathlib import Path

EXPERIMENT_NAME = "FraudLens"


def _estimator(model):
    return model.named_steps["model"] if hasattr(model, "named_steps") else model


def _params(model) -> dict[str, str]:
    return {key: str(value) for key, value in _estimator(model).get_params().items() if value is not None}


def _metric_set(prefix: str, metrics: dict) -> dict[str, float]:
    values = {
        f"{prefix}_average_precision": metrics["average_precision"],
        f"{prefix}_roc_auc": metrics["roc_auc"],
        f"{prefix}_precision_at_200": metrics["precision_at_k"],
        f"{prefix}_true_positives_in_top_200": float(metrics["true_positives"]),
        f"{prefix}_precision_at_threshold": metrics["precision"],
        f"{prefix}_recall_at_threshold": metrics["recall"],
        f"{prefix}_f1_at_threshold": metrics["f1"],
    }
    if metrics["recall_at_k"] is not None:
        values[f"{prefix}_recall_at_200"] = metrics["recall_at_k"]
    return values


def log_experiment(metadata: dict, models: dict, report: dict, artifact_dir: str | Path) -> str | None:
    """Log one run per candidate and a selected run; returns the selected run ID.

    Candidate runs hold validation metrics only. Held-out test metrics are logged once, in the
    selected run, after model selection. Returns None when MLflow is not installed.
    """
    try:
        import mlflow
    except ImportError:
        return None

    artifacts = Path(artifact_dir)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (Path.cwd() / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    selected_name = metadata["selected_model"]

    shared = {
        "random_state": metadata["random_state"],
        "split_protocol": "stratified 75/25 train/test; validation 20% of train",
        "train_rows": metadata["train_rows"],
        "validation_rows": metadata["validation_rows"],
        "test_rows": metadata["test_rows"],
        "preprocessing": "Amount scaled with StandardScaler fitted on training rows (logistic only); V1-V28 unchanged",
        "imbalance_strategy": "class weighting (no resampling)",
        "dataset_sha256": metadata["dataset_fingerprint"],
        "dataset_file": "creditcard.csv",
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"candidate-{name}"):
            mlflow.set_tag("role", "candidate")
            mlflow.set_tag("selected", str(name == selected_name).lower())
            mlflow.log_param("model_type", name)
            mlflow.log_params(shared)
            mlflow.log_params({f"param_{key}": value for key, value in _params(model).items()})
            mlflow.log_metrics(_metric_set("validation", metadata["validation_metrics"][name]))

    with mlflow.start_run(run_name=f"selected-{selected_name}") as run:
        mlflow.set_tag("role", "selected_final")
        mlflow.set_tag("evaluation_split", report["evaluation_split"])
        mlflow.log_param("model_type", selected_name)
        mlflow.log_param("selection_basis", report["selection_basis"])
        mlflow.log_params(shared)
        mlflow.log_metrics(_metric_set("validation", metadata["validation_metrics"][selected_name]))
        mlflow.log_metrics(_metric_set("test", metadata["test_metrics"]))
        mlflow.log_metric(
            "test_pr_auc_trapezoidal", report["models"][selected_name]["pr_curve"]["pr_auc_trapezoidal"]
        )
        mlflow.log_metric("test_dummy_baseline_accuracy", report["dummy_baseline"]["accuracy"])
        for filename in [
            "model.joblib",
            "metadata.json",
            "evaluation_report.json",
            "top_200_test_queue.csv",
            "eda_summary.json",
        ]:
            if (artifacts / filename).exists():
                mlflow.log_artifact(str(artifacts / filename))
        if (artifacts / "plots").exists():
            mlflow.log_artifacts(str(artifacts / "plots"), artifact_path="plots")
        return run.info.run_id


def list_runs() -> list[dict]:
    """Summarize logged runs for the dashboard without exposing the tracking store location."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed; install mlflow to view experiment runs.") from exc
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (Path.cwd() / "mlruns").resolve().as_uri()
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("No FraudLens MLflow experiment found. Run `python scripts/train.py` first.")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=20)
    keep = ("average_precision", "roc_auc", "precision_at_200", "recall_at_200", "true_positives_in_top_200")
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "role": run.data.tags.get("role"),
            "selected": run.data.tags.get("selected") == "true" or run.data.tags.get("role") == "selected_final",
            "status": run.info.status,
            "metrics": {key: value for key, value in run.data.metrics.items() if key.endswith(keep)},
        }
        for run in runs
    ]
