from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from .validation import FEATURE_COLUMNS, load_dataset

KEY_FEATURES = ["V14", "V17", "V12", "V10"]
TIME_BIN_SECONDS = 7200


def _describe(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def summarize_dataset(path: str | Path) -> dict:
    return _summarize(load_dataset(path))


def _summarize(frame: pd.DataFrame) -> dict:
    counts = frame["Class"].value_counts().sort_index()
    fraud = frame[frame["Class"] == 1]
    legit = frame[frame["Class"] == 0]

    separation = []
    for column in FEATURE_COLUMNS:
        auc = float(roc_auc_score(frame["Class"], frame[column]))
        separation.append(
            {
                "feature": column,
                "single_feature_auc": auc,
                "separation": max(auc, 1 - auc),
                "fraud_median": float(fraud[column].median()),
                "legitimate_median": float(legit[column].median()),
            }
        )
    separation.sort(key=lambda item: item["separation"], reverse=True)
    ranks = {item["feature"]: position + 1 for position, item in enumerate(separation)}

    correlations = frame[FEATURE_COLUMNS + ["Class"]].corr()["Class"].drop("Class")
    top_correlated = correlations.reindex(correlations.abs().sort_values(ascending=False).index).head(10)

    amount_test = mannwhitneyu(fraud["Amount"], legit["Amount"], alternative="two-sided")
    fraud_amount, legit_amount = _describe(fraud["Amount"]), _describe(legit["Amount"])
    median_lower = fraud_amount["median"] < legit_amount["median"]
    mean_lower = fraud_amount["mean"] < legit_amount["mean"]

    bins = (frame["Time"] // TIME_BIN_SECONDS).astype(int)
    by_bin = frame.groupby(bins)["Class"].agg(["size", "sum"])
    time_bins = [
        {
            "bin_start_seconds": int(index * TIME_BIN_SECONDS),
            "transactions": int(row["size"]),
            "frauds": int(row["sum"]),
            "fraud_rate": float(row["sum"] / row["size"]),
        }
        for index, row in by_bin.iterrows()
    ]
    first_day = frame[frame["Time"] < 86400]
    second_day = frame[(frame["Time"] >= 86400) & (frame["Time"] < 172800)]
    first_counts = first_day.groupby((first_day["Time"] // 3600).astype(int)).size()
    second_counts = second_day.groupby(((second_day["Time"] - 86400) // 3600).astype(int)).size()
    aligned = pd.concat([first_counts, second_counts], axis=1).dropna()
    repeat_corr = float(aligned.corr().iloc[0, 1]) if len(aligned) > 2 else None

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "feature_columns": FEATURE_COLUMNS,
        "dtypes": {str(key): int(value) for key, value in frame.dtypes.astype(str).value_counts().items()},
        "fraud_count": int(counts.get(1, 0)),
        "legitimate_count": int(counts.get(0, 0)),
        "fraud_prevalence": float(frame["Class"].mean()),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "all_negative_accuracy": float(counts.get(0, 0) / len(frame)),
        "amount": {key: float(value) for key, value in frame["Amount"].describe().items()},
        "time": {key: float(value) for key, value in frame["Time"].describe().items()},
        "amount_by_class": {"fraud": fraud_amount, "legitimate": legit_amount},
        "amount_hypothesis": {
            "hypothesis": "Fraudulent transactions tend to have smaller amounts.",
            "fraud_median_lower": bool(median_lower),
            "fraud_mean_lower": bool(mean_lower),
            "mann_whitney_p_value": float(amount_test.pvalue),
            "finding": (
                f"Median Amount: fraud {fraud_amount['median']:.2f} vs legitimate {legit_amount['median']:.2f}; "
                f"mean Amount: fraud {fraud_amount['mean']:.2f} vs legitimate {legit_amount['mean']:.2f}. "
                + (
                    "The typical (median) fraud amount is smaller, "
                    if median_lower
                    else "The typical (median) fraud amount is not smaller, "
                )
                + (
                    "and the mean is also smaller."
                    if mean_lower
                    else "but the mean is larger because a few large fraud amounts pull it up."
                )
            ),
        },
        "key_features": {
            name: {
                "fraud": _describe(fraud[name]),
                "legitimate": _describe(legit[name]),
                "separation_rank_of_30": ranks[name],
                "single_feature_auc": next(i["single_feature_auc"] for i in separation if i["feature"] == name),
            }
            for name in KEY_FEATURES
        },
        "feature_separation_top10": separation[:10],
        "top_correlations_with_class": {key: float(value) for key, value in top_correlated.items()},
        "time_bins_2h_elapsed": time_bins,
        "time_note": (
            "Time is elapsed seconds from the first transaction, not a clock time or calendar date. "
            "The dataset spans about two elapsed days."
        ),
        "time_first_vs_second_24h_count_correlation": repeat_corr,
    }


def plot_eda(frame: pd.DataFrame, summary: dict, output_dir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fraud = frame[frame["Class"] == 1]
    legit = frame[frame["Class"] == 0]
    files = []

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(out / name, dpi=110)
        plt.close(figure)
        files.append(name)

    counts = [summary["legitimate_count"], summary["fraud_count"]]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Legitimate", "Fraud"], counts, color=["#38bdf8", "#f87171"])
    axis.set_yscale("log")
    axis.set_ylabel("Transactions (log scale)")
    axis.set_title(f"Class balance: fraud prevalence {summary['fraud_prevalence']:.4%}")
    for position, value in enumerate(counts):
        axis.text(position, value, f"{value:,}", ha="center", va="bottom")
    save(figure, "eda_class_balance.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-1, np.log10(frame["Amount"].max() + 1), 50)
    axes[0].hist(legit["Amount"] + 0.1, bins=bins, alpha=0.6, density=True, label="Legitimate", color="#38bdf8")
    axes[0].hist(fraud["Amount"] + 0.1, bins=bins, alpha=0.6, density=True, label="Fraud", color="#f87171")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Amount + 0.1 (log scale)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution by class")
    axes[0].legend()
    axes[1].boxplot([legit["Amount"], fraud["Amount"]], tick_labels=["Legitimate", "Fraud"], showfliers=False)
    axes[1].set_title("Amount by class (outliers hidden)")
    save(figure, "eda_amount_by_class.png")

    figure, axis = plt.subplots(figsize=(9, 4))
    edges = np.arange(0, frame["Time"].max() + TIME_BIN_SECONDS, TIME_BIN_SECONDS)
    axis.hist(legit["Time"], bins=edges, alpha=0.6, density=True, label="Legitimate", color="#38bdf8")
    axis.hist(fraud["Time"], bins=edges, alpha=0.6, density=True, label="Fraud", color="#f87171")
    axis.set_xlabel("Time (elapsed seconds since first transaction; not clock time)")
    axis.set_ylabel("Density")
    axis.set_title("Elapsed-time distribution by class")
    axis.legend()
    save(figure, "eda_time_by_class.png")

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, name in zip(axes.ravel(), KEY_FEATURES):
        low, high = frame[name].quantile([0.001, 0.999])
        edges = np.linspace(low, high, 60)
        axis.hist(legit[name].clip(low, high), bins=edges, alpha=0.6, density=True, label="Legitimate", color="#38bdf8")
        axis.hist(fraud[name].clip(low, high), bins=edges, alpha=0.6, density=True, label="Fraud", color="#f87171")
        rank = summary["key_features"][name]["separation_rank_of_30"]
        axis.set_title(f"{name} (single-feature separation rank {rank}/30)")
        axis.legend()
    save(figure, "eda_key_features.png")

    names = list(summary["top_correlations_with_class"])
    values = [summary["top_correlations_with_class"][name] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(names[::-1], values[::-1], color=["#f87171" if v > 0 else "#38bdf8" for v in values[::-1]])
    axis.set_xlabel("Pearson correlation with Class")
    axis.set_title("Top 10 features by |correlation| with fraud label")
    save(figure, "eda_class_correlations.png")

    top = [item["feature"] for item in summary["feature_separation_top10"]]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(frame[top].corr(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(top)), top, rotation=45)
    axis.set_yticks(range(len(top)), top)
    figure.colorbar(image)
    axis.set_title("Correlation among the 10 most class-separating features")
    save(figure, "eda_feature_correlation.png")
    return files


def run_eda(path: str | Path, artifact_dir: str | Path) -> dict:
    frame = load_dataset(path)
    summary = _summarize(frame)
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary["plots"] = plot_eda(frame, summary, out / "plots")
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_summary(path: str | Path, output_path: str | Path) -> dict:
    """Write the EDA summary and plots into the directory containing output_path."""
    return run_eda(path, Path(output_path).parent)
