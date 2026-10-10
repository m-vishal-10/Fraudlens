from __future__ import annotations

import json
from pathlib import Path

import joblib

from .validation import FEATURE_COLUMNS


class ArtifactUnavailableError(RuntimeError):
    """Raised when the deployable model artifact cannot be safely loaded."""


def load_artifact(artifact_dir: str | Path):
    directory = Path(artifact_dir)
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ArtifactUnavailableError(
            f"Model artifacts are unavailable in {directory}. Run `python -m ml.train` first."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ArtifactUnavailableError("Model artifact feature schema does not match the API schema.")
        return joblib.load(model_path), metadata
    except ArtifactUnavailableError:
        raise
    except Exception as exc:
        raise ArtifactUnavailableError(f"Model artifacts are corrupt or unreadable: {exc}") from exc
