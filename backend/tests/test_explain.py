import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.explain import contributions, summarize_reasons, top_reasons
from ml.preprocessing import build_logistic_pipeline
from ml.validation import FEATURE_COLUMNS


def synthetic_data(rows=400):
    generator = np.random.default_rng(0)
    frame = pd.DataFrame(generator.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    labels = (frame["V14"] + 0.5 * frame["V4"] > 1.2).astype(int)
    return frame, labels


def logit(probabilities):
    return np.log(probabilities / (1 - probabilities))


def test_xgboost_contributions_sum_to_raw_score():
    frame, labels = synthetic_data()
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42).fit(frame, labels)
    values, bias = contributions(model, frame)
    expected = logit(np.clip(model.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6))
    assert np.allclose(values.sum(axis=1) + bias, expected, atol=1e-3)


def test_logistic_contributions_sum_to_raw_score():
    frame, labels = synthetic_data()
    model = build_logistic_pipeline(LogisticRegression(max_iter=2000)).fit(frame, labels)
    values, bias = contributions(model, frame)
    expected = model.decision_function(frame)
    assert np.allclose(values.sum(axis=1) + bias, expected, atol=1e-6)


def test_top_reasons_are_ordered_and_use_real_fields():
    frame, labels = synthetic_data()
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42).fit(frame, labels)
    reasons = top_reasons(model, frame.head(3), top_n=4)
    assert len(reasons) == 3
    for row in reasons:
        assert len(row) == 4
        magnitudes = [abs(item["contribution"]) for item in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(item["feature"] in FEATURE_COLUMNS for item in row)
    assert "Class" not in {item["feature"] for row in reasons for item in row}


def test_summary_uses_cautious_language():
    text = summarize_reasons(
        [{"feature": "V14", "direction": "increases"}, {"feature": "V4", "direction": "decreases"}]
    )
    assert "V14" in text and "V4" not in text
    assert "not proof of fraud" in text
