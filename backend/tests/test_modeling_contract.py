import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.data_loader import load_features_and_target
from ml.preprocessing import build_logistic_pipeline
from ml.train import split_data
from ml.validation import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    DatasetValidationError,
    load_dataset,
    validate_labeled_frame,
)


def labeled_frame(rows=2000, fraud=40, seed=1):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(generator.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["Amount"] = generator.gamma(2.0, 30.0, size=rows)
    frame["Class"] = 0
    frame.loc[generator.choice(rows, fraud, replace=False), "Class"] = 1
    return frame


def test_split_is_stratified_disjoint_and_uses_75_25():
    frame = labeled_frame()
    X, y = frame[FEATURE_COLUMNS], frame["Class"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    assert len(X_test) == 500
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert not (set(X_train.index) & set(X_test.index))
    assert not (set(X_val.index) & set(X_test.index))
    assert not (set(X_train.index) & set(X_val.index))
    assert y_test.sum() == 10 and y_train.sum() + y_val.sum() == 30
    assert abs(y_train.mean() - y.mean()) < 0.005


def test_split_is_reproducible():
    frame = labeled_frame()
    first = split_data(frame[FEATURE_COLUMNS], frame["Class"])
    second = split_data(frame[FEATURE_COLUMNS], frame["Class"])
    assert first[2].index.equals(second[2].index)


def test_target_is_excluded_from_features(tmp_path):
    path = tmp_path / "data.csv"
    labeled_frame()[EXPECTED_COLUMNS].to_csv(path, index=False)
    X, y = load_features_and_target(path)
    assert "Class" not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}


def test_scaler_is_fitted_on_training_rows_only():
    frame = labeled_frame()
    X, y = frame[FEATURE_COLUMNS], frame["Class"]
    X_train, _, X_test, y_train, _, _ = split_data(X, y)
    pipeline = build_logistic_pipeline(LogisticRegression(max_iter=200)).fit(X_train, y_train)
    scaler = pipeline.named_steps["preprocess"].named_transformers_["amount_scaler"]
    assert scaler.mean_[0] == pytest.approx(X_train["Amount"].mean())
    assert scaler.mean_[0] != pytest.approx(X["Amount"].mean())


def test_schema_validation_rejects_bad_datasets(tmp_path):
    good = labeled_frame()[EXPECTED_COLUMNS]
    validate_labeled_frame(good)

    with pytest.raises(DatasetValidationError):
        validate_labeled_frame(good.drop(columns=["V5"]))
    with pytest.raises(DatasetValidationError):
        validate_labeled_frame(good.assign(Class=2))
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")
