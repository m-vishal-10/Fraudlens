from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .data_loader import dataset_fingerprint, load_features_and_target
from .evaluate import classification_metrics
from .mlflow_utils import log_experiment
from .preprocessing import build_logistic_pipeline
from .report import build_evaluation_report
from .validation import FEATURE_COLUMNS


def split_data(X, y):
    """Stratified 75/25 train/test split, then a validation split taken from the training part only."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_models(data_path: str, artifact_dir: str) -> dict:
    X, y = load_features_and_target(data_path)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)
    candidates = {
        "logistic_regression": build_logistic_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            random_state=42,
            n_jobs=2,
        ),
    }
    validation = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_validation)[:, 1]
        validation[name] = classification_metrics(y_validation, scores)
        fitted[name] = model
    selected_name = max(
        validation,
        key=lambda name: (
            validation[name]["average_precision"],
            validation[name]["precision_at_k"] or 0.0,
        ),
    )
    selected = fitted[selected_name]
    test_scores = selected.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_scores)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, output / "model.joblib")
    metadata = {
        "selected_model": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "Class",
        "random_state": 42,
        "split": {"train_fraction": 0.75, "test_fraction": 0.25, "validation_from_train": 0.2},
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "class_counts": {
            name: {"legitimate": int((part == 0).sum()), "fraud": int((part == 1).sum())}
            for name, part in {
                "train": y_train,
                "validation": y_validation,
                "test": y_test,
            }.items()
        },
        "validation_metrics": validation,
        "test_metrics": test_metrics,
        "dataset_fingerprint": dataset_fingerprint(data_path),
        "dataset_path": str(Path(data_path).resolve()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = build_evaluation_report(fitted, selected_name, X_test, y_test, output)
    metadata["mlflow_run_id"] = log_experiment(metadata, fitted, report, output)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train_models(args.data, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
