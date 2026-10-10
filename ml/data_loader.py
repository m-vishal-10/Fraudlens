from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .validation import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset


def add_transaction_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "transaction_id", [f"tx-{index:06d}" for index in range(len(result))])
    return result


def dataset_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features_and_target(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = load_dataset(path)
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].copy()
