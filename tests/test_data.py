import json

import numpy as np
import pandas as pd
import pytest

from demo.cnn_schema import CNN_FEATURE_COLUMNS
from webapp.server import DATA, FAULTS

PARQUET_PATH = DATA / "twin_library.parquet"
ANCHORS_PATH = DATA / "anchors.json"

@pytest.fixture(scope="module")
def twin_df():
    assert PARQUET_PATH.is_file(), f"Không tìm thấy file: {PARQUET_PATH}"
    return pd.read_parquet(PARQUET_PATH)

def test_twin_library_schema_and_types(twin_df):
    for col in CNN_FEATURE_COLUMNS:
        assert col in twin_df.columns, f"Thiếu cột cảm biến: {col}"
        assert (
            twin_df[col].dtype == np.float32
        ), f"Cột {col} phải có kiểu float32 để tối ưu dung lượng!"

    # Không được chứa giá trị NaN hoặc Inf
    assert (
        twin_df[CNN_FEATURE_COLUMNS].isna().sum().sum() == 0
    ), "Dữ liệu chứa giá trị rỗng NaN!"
    assert np.all(
        np.isfinite(twin_df[CNN_FEATURE_COLUMNS].to_numpy())
    ), "Dữ liệu chứa giá trị vô cực Inf!"

def test_twin_library_fault_coverage(twin_df):
    unique_faults = set(twin_df["fault"].unique())
    expected_faults = {"normal", *FAULTS}
    assert (
        expected_faults == unique_faults
    ), f"Dữ liệu thiếu loại lỗi! Hiện có: {unique_faults}"

    normal_sev = twin_df.loc[twin_df["fault"] == "normal", "severity"].unique()
    assert list(normal_sev) == [0]

    fault_df = twin_df[twin_df["fault"] != "normal"]
    assert set(fault_df["severity"].unique()).issubset({1, 2, 3, 4})

def test_anchors_json_consistency():
    assert ANCHORS_PATH.is_file(), f"Không tìm thấy file: {ANCHORS_PATH}"
    anchors = json.loads(ANCHORS_PATH.read_text())

    assert "normal" in anchors
    assert "0" in anchors["normal"]
    for fault in FAULTS:
        assert fault in anchors, f"anchors.json thiếu cấu hình cho lỗi: {fault}"