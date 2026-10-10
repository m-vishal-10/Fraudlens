import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.eda import _summarize
from ml.preprocessing import build_logistic_pipeline
from ml.report import build_evaluation_report, queue_statement
from ml.validation import FEATURE_COLUMNS


def labeled_frame(rows=1200, seed=0):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(generator.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["Amount"] = generator.gamma(2.0, 30.0, size=rows)
    frame["Time"] = np.sort(generator.uniform(0, 172000, size=rows))
    labels = (frame["V14"] + 0.4 * frame["V4"] > 1.8).astype(int)
    return frame, labels


def test_queue_statement_reports_counts_and_capture():
    text = queue_statement({"k": 200, "true_positives": 106, "recall_at_k": 0.8617})
    assert "106 were actual fraud cases" in text and "86.2%" in text


def test_queue_statement_marks_recall_unavailable_without_positives():
    text = queue_statement({"k": 5, "true_positives": 0, "recall_at_k": None})
    assert "unavailable" in text


def test_evaluation_report_outputs_and_dummy_baseline(tmp_path):
    frame, labels = labeled_frame()
    train, test = frame.iloc[:900], frame.iloc[900:]
    models = {
        "logistic_regression": build_logistic_pipeline(LogisticRegression(max_iter=500)).fit(train, labels.iloc[:900]),
        "xgboost": XGBClassifier(n_estimators=15, max_depth=3, random_state=42).fit(train, labels.iloc[:900]),
    }
    y_test = labels.iloc[900:]
    report = build_evaluation_report(models, "xgboost", test, y_test, tmp_path)

    assert report["evaluation_split"] == "held_out_test"
    assert report["dummy_baseline"]["recall"] == 0
    assert report["dummy_baseline"]["accuracy"] == 1 - y_test.mean()
    assert set(report["models"]) == {"logistic_regression", "xgboost"}
    assert (tmp_path / "plots" / "pr_curve.png").exists()
    assert (tmp_path / "plots" / "confusion_matrix.png").exists()

    queue = pd.read_csv(tmp_path / "top_200_test_queue.csv")
    assert len(queue) == min(200, len(test))
    assert queue["risk_score"].is_monotonic_decreasing
    assert report["models"]["xgboost"]["metrics"]["true_positives"] == int(
        queue["actual_class_retrospective_label"].sum()
    )


def test_eda_tests_amount_hypothesis_from_data_not_assumption():
    frame, labels = labeled_frame(rows=2000)
    frame["Class"] = labels
    frame.loc[frame["Class"] == 1, "Amount"] = 500.0
    summary = _summarize(frame)
    assert summary["amount_hypothesis"]["fraud_median_lower"] is False
    assert summary["fraud_count"] == int(labels.sum())
    assert set(summary["key_features"]) == {"V14", "V17", "V12", "V10"}
    assert summary["all_negative_accuracy"] == 1 - labels.mean()
