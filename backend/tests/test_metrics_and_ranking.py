import pandas as pd

from ml.evaluate import dummy_metrics, top_k_metrics
from ml.ranking import rank_alerts
from ml.validation import FEATURE_COLUMNS, DatasetValidationError, validate_inference_frame


def features(rows=3):
    return pd.DataFrame([{column: float(index) for index, column in enumerate(FEATURE_COLUMNS)} for _ in range(rows)])


def test_top_k_and_no_positive_recall():
    result = top_k_metrics([0, 1, 0], [0.1, 0.9, 0.8], capacity=2)
    assert result["k"] == 2
    assert result["true_positives"] == 1
    assert result["precision_at_k"] == 0.5
    assert top_k_metrics([0, 0], [0.1, 0.2])["recall_at_k"] is None


def test_dummy_baseline_has_zero_recall():
    result = dummy_metrics([0, 0, 1])
    assert result["accuracy"] == 2 / 3
    assert result["recall"] == 0


def test_rank_is_deterministic_and_capped():
    result = rank_alerts(features(), [0.2, 0.9, 0.9], capacity=2)
    assert len(result) == 2
    assert result["rank"].tolist() == [1, 2]
    assert result["risk_score"].tolist() == [0.9, 0.9]
    assert result["transaction_id"].tolist() == ["tx-000001", "tx-000002"]


def test_target_is_rejected_from_inference():
    invalid = features(1)
    invalid["Class"] = 0
    try:
        validate_inference_frame(invalid)
    except DatasetValidationError as exc:
        assert "Class" in str(exc)
    else:
        raise AssertionError("Class should not be accepted for inference")
