from __future__ import annotations

import pandas as pd

from .validation import FEATURE_COLUMNS, validate_inference_frame

CAPACITY = 200


def rank_alerts(features: pd.DataFrame, scores, capacity: int = CAPACITY) -> pd.DataFrame:
    validate_inference_frame(features)
    if capacity < 1:
        raise ValueError("capacity must be positive.")
    result = features.reset_index(drop=True).copy()
    result.insert(0, "transaction_id", [f"tx-{index:06d}" for index in range(len(result))])
    result["risk_score"] = pd.Series(scores, dtype="float64").reset_index(drop=True)
    result = result.sort_values(
        ["risk_score", "transaction_id"], ascending=[False, True], kind="mergesort"
    ).head(min(capacity, len(result))).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    result["risk_category"] = result["risk_score"].map(
        lambda score: "High" if score >= 0.75 else "Medium" if score >= 0.35 else "Review"
    )
    result["analyst_note"] = result.apply(
        lambda row: (
            f"Priority #{int(row['rank'])} | Risk score: {row['risk_score']:.4f} | "
            f"Amount: {row['Amount']:.2f} | Elevated model-assigned fraud risk; "
            "review transaction context and verify cardholder activity."
        ),
        axis=1,
    )
    return result[
        ["rank", "transaction_id", "risk_score", "Amount", "Time", "risk_category", "analyst_note"]
    ]
