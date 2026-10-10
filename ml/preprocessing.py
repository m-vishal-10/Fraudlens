from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("amount_scaler", StandardScaler(), ["Amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
