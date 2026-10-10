import pandas as pd
import pytest

from ml.validation import FEATURE_COLUMNS, DatasetValidationError, validate_inference_frame


def test_finite_feature_schema_is_valid():
    frame = pd.DataFrame([{column: 1.0 for column in FEATURE_COLUMNS}])
    validate_inference_frame(frame)


def test_nan_is_rejected():
    frame = pd.DataFrame([{column: 1.0 for column in FEATURE_COLUMNS}])
    frame.loc[0, "Amount"] = float("nan")
    with pytest.raises(DatasetValidationError):
        validate_inference_frame(frame)
