from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from .evaluate import classification_metrics, dummy_metrics

CAPACITY = 200
CURVE_POINTS = 400
THRESHOLD = 0.5


def _curve(labels, scores) -> dict:
    precision, recall, _ = precision_recall_curve(labels, scores)
    keep = np.unique(np.linspace(0, len(recall) - 1, CURVE_POINTS).astype(int))
    return {
        "precision": [float(value) for value in precision[keep]],
        "recall": [float(value) for value in recall[keep]],
        "pr_auc_trapezoidal": float(auc(recall, precision)),
    }


def queue_statement(metrics: dict) -> str:
    k, tp, recall = metrics["k"], metrics["true_positives"], metrics["recall_at_k"]
    capture = f"{recall:.1%}" if recall is not None else "unavailable (no fraud cases in this batch)"
    return (
        f"Of the {k} transactions selected for review, {tp} were actual fraud cases. "
        f"The queue captured {capture} of all fraud cases in the evaluated batch."
    )


def build_evaluation_report(models: dict, selected: str, X_test, y_test, artifact_dir: str | Path) -> dict:
    """Retrospective report on the held-out test split.

    Called only after model selection on validation data. Both models are scored on the test
    split for reporting; the test results are never used to choose between them.
    """
    out = Path(artifact_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(y_test)
    prevalence = float(labels.mean())

    scores = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
    model_reports = {}
    for name, values in scores.items():
        metrics = classification_metrics(labels, values, threshold=THRESHOLD)
        model_reports[name] = {
            "metrics": metrics,
            "pr_curve": _curve(labels, values),
            "queue_statement": queue_statement(metrics),
        }

    order = np.lexsort((np.asarray(X_test.index), -scores[selected]))[: min(CAPACITY, len(labels))]
    queue = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "transaction_id": [f"tx-{index:06d}" for index in X_test.index[order]],
            "risk_score": scores[selected][order],
            "Amount": X_test["Amount"].to_numpy()[order],
            "Time": X_test["Time"].to_numpy()[order],
            "actual_class_retrospective_label": labels[order],
        }
    )
    queue.to_csv(out / "top_200_test_queue.csv", index=False)

    report = {
        "evaluation_split": "held_out_test",
        "note": (
            "Retrospective evaluation on a stratified random 25% test split that was not used for "
            "training or model selection. A benchmark, not evidence of future daily performance."
        ),
        "rows": int(len(labels)),
        "fraud_count": int(labels.sum()),
        "fraud_prevalence": prevalence,
        "threshold_for_confusion_matrix": THRESHOLD,
        "selected_model": selected,
        "selection_basis": "validation Average Precision, validation precision@200 as tie-breaker",
        "capacity": CAPACITY,
        "dummy_baseline": dummy_metrics(labels),
        "models": model_reports,
        "selected_queue_statement": model_reports[selected]["queue_statement"],
        "queue_csv": "top_200_test_queue.csv",
    }
    _plot(report, plots)
    report["plots"] = ["pr_curve.png", "confusion_matrix.png"]
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _plot(report: dict, plots: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"logistic_regression": "#38bdf8", "xgboost": "#f87171"}
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, item in report["models"].items():
        curve = item["pr_curve"]
        ap = item["metrics"]["average_precision"]
        axis.plot(curve["recall"], curve["precision"], color=colors.get(name), label=f"{name} (AP {ap:.3f})")
    axis.axhline(report["fraud_prevalence"], color="gray", linestyle="--", label="Fraud prevalence baseline")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curve (held-out test set)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "pr_curve.png", dpi=110)
    plt.close(figure)

    matrix = np.array(report["models"][report["selected_model"]]["metrics"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:,}", ha="center", va="center", color="black")
    axis.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
    axis.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    axis.set_title(f"{report['selected_model']} at threshold {report['threshold_for_confusion_matrix']}")
    figure.tight_layout()
    figure.savefig(plots / "confusion_matrix.png", dpi=110)
    plt.close(figure)
