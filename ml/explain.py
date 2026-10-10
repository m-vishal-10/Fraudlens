from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from .validation import FEATURE_COLUMNS


def contributions(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row feature contributions to the model's raw (log-odds) score, plus the bias term.

    For every row, contributions.sum() + bias equals the model's raw score before the sigmoid.
    """
    frame = features[FEATURE_COLUMNS]
    if isinstance(model, xgb.XGBClassifier):
        values = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)
        return values[:, :-1], values[:, -1]
    if isinstance(model, Pipeline) and hasattr(model.named_steps.get("model"), "coef_"):
        preprocess = model.named_steps["preprocess"]
        classifier = model.named_steps["model"]
        transformed = np.asarray(preprocess.transform(frame))
        names = list(preprocess.get_feature_names_out())
        per_feature = transformed * classifier.coef_[0]
        ordered = per_feature[:, [names.index(column) for column in FEATURE_COLUMNS]]
        return ordered, np.full(len(frame), float(classifier.intercept_[0]))
    raise TypeError(f"Explanations are not supported for model type {type(model).__name__}.")


def top_reasons(model, features: pd.DataFrame, top_n: int = 5) -> list[list[dict]]:
    """Return the top_n features per row ranked by absolute contribution to the score."""
    values, _ = contributions(model, features)
    result: list[list[dict]] = []
    for row_index in range(len(features)):
        order = np.argsort(-np.abs(values[row_index]), kind="stable")[:top_n]
        reasons = []
        for column_index in order:
            feature = FEATURE_COLUMNS[column_index]
            contribution = float(values[row_index, column_index])
            direction = "increases" if contribution > 0 else "decreases"
            value = float(features.iloc[row_index][feature])
            reasons.append(
                {
                    "feature": feature,
                    "value": value,
                    "contribution": contribution,
                    "direction": direction,
                    "description": (
                        f"{feature} = {value:.3f} {direction} the risk score "
                        f"({contribution:+.2f} log-odds)"
                    ),
                }
            )
        result.append(reasons)
    return result


def summarize_reasons(reasons: list[dict], limit: int = 3) -> str:
    upward = [item["feature"] for item in reasons if item["direction"] == "increases"][:limit]
    if not upward:
        return "No feature strongly increased this score."
    return (
        "Largest model-score drivers: "
        + ", ".join(upward)
        + ". This explains model behavior, not proof of fraud; V1-V28 are anonymized components."
    )
