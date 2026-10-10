from __future__ import annotations

import pandas as pd

from ml.artifacts import ArtifactUnavailableError, load_artifact
from ml.explain import summarize_reasons, top_reasons
from ml.ranking import rank_alerts
from ml.validation import FEATURE_COLUMNS, validate_inference_frame


class ModelService:
    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.model = None
        self.metadata = None

    def load(self) -> None:
        self.model, self.metadata = load_artifact(self.artifact_dir)

    @property
    def available(self) -> bool:
        return self.model is not None

    def ensure_loaded(self) -> None:
        if not self.available:
            self.load()

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        validate_inference_frame(features)
        self.ensure_loaded()
        scores = self.model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
        ranked = rank_alerts(features, scores)
        # Explain only the queued rows; transaction IDs encode the original row position.
        positions = ranked["transaction_id"].str[3:].astype(int).tolist()
        queued = features.iloc[positions].reset_index(drop=True)
        ranked["reasons"] = top_reasons(self.model, queued)
        ranked["analyst_note"] = [
            f"{note} {summarize_reasons(reasons)}"
            for note, reasons in zip(ranked["analyst_note"], ranked["reasons"])
        ]
        return ranked
