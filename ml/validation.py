from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


class DatasetValidationError(ValueError):
    """Raised when a dataset does not match the supported contract."""


def validate_labeled_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
        raise DatasetValidationError(
            f"Expected columns {EXPECTED_COLUMNS}; missing={missing}, unexpected={unexpected}."
        )
    if frame.empty:
        raise DatasetValidationError("The dataset is empty.")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise DatasetValidationError("Feature columns contain missing values.")
    if not frame[FEATURE_COLUMNS].map(lambda value: pd.api.types.is_number(value)).all().all():
        raise DatasetValidationError("All feature values must be numeric.")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise DatasetValidationError("Class must contain only 0 and 1.")


def validate_inference_frame(frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    if TARGET_COLUMN in columns:
        raise DatasetValidationError("Inference input must not include the Class target column.")
    if columns != FEATURE_COLUMNS:
        missing = sorted(set(FEATURE_COLUMNS) - set(columns))
        unexpected = sorted(set(columns) - set(FEATURE_COLUMNS))
        raise DatasetValidationError(
            f"Expected inference columns {FEATURE_COLUMNS}; missing={missing}, unexpected={unexpected}."
        )
    if frame.empty:
        raise DatasetValidationError("The inference batch is empty.")
    if frame.isna().any().any():
        raise DatasetValidationError("Inference features contain missing values.")
    if not frame.map(lambda value: pd.api.types.is_number(value)).all().all():
        raise DatasetValidationError("Inference features must be numeric.")
    if not frame.map(lambda value: pd.api.types.is_number(value) and np.isfinite(value)).all().all():
        raise DatasetValidationError("Inference features must be finite numeric values.")


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    frame = pd.read_csv(dataset_path)
    validate_labeled_frame(frame)
    return frame
