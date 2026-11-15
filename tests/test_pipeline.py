from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from demo.cnn_schema import (
    CNN_FEATURE_COLUMNS,
    CNN_INPUT_CHANNELS,
    CNN_MODEL_FEATURE_COLUMNS,
)
from demo.pipeline import (
    STRIDE,
    WINDOW,
    cut_windows,
    diagnose_frame,
    load_bundle,
    predict_windows,
    prepare_model_frame,
)

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_SPLIT_PATH = TEST_DATA_DIR / "test_split.parquet"
MIN_ACCURACY = 0.5095
MIN_MACRO_F1 = 0.48


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()

def test_model_forward_pass_with_dummy_tensor(bundle):
    model = bundle["model"]
    device = bundle["device"]

    batch_size = 2
    dummy_windows = np.random.randn(batch_size, CNN_INPUT_CHANNELS, WINDOW).astype(np.float32)

    preds, probas = predict_windows(model, dummy_windows, device)

    assert len(preds) == batch_size
    assert probas.shape == (batch_size, 8)  
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(batch_size), atol=1e-5)

def test_cut_windows_shape_contract():
    n_samples = 200
    fake_data = {col: np.random.randn(n_samples) for col in CNN_MODEL_FEATURE_COLUMNS}
    fake_data["time"] = np.arange(n_samples) * 10.0
    fake_data["source_file"] = "synthetic_stream"
    df = pd.DataFrame(fake_data)

    xs, meta = cut_windows(df, window=WINDOW, stride=64)

    assert len(xs) > 0
    windows_arr = np.array(xs)
    assert windows_arr.shape[1] == CNN_INPUT_CHANNELS  
    assert windows_arr.shape[2] == WINDOW
    assert len(meta) == len(xs)

def test_diagnose_frame_end_to_end(bundle):
    """Tầng 3: Kiểm tra toàn bộ pipeline chẩn đoán từ DataFrame thô đến kết quả sức khỏe."""
    from webapp.server import DATA

    # 1. Lấy 150 dòng và ép kiểu float64 để tính toán ma trận chính xác
    lib = pd.read_parquet(DATA / "twin_library.parquet")
    sample_stream = lib.iloc[:150][CNN_FEATURE_COLUMNS].reset_index(drop=True).astype("float64")
    sample_stream["time"] = np.arange(150) * 10.0

    # 2. Chạy chẩn đoán qua toàn bộ pipeline
    diag = diagnose_frame(sample_stream, bundle)

    # 3. Kiểm tra kết quả trả về
    assert diag.n_windows > 0
    assert 0.0 <= diag.health_ratio <= 1.0
    assert diag.t_span_s > 0
    assert isinstance(diag.segments, list)
    assert isinstance(diag.windows, list)


def test_tc06_model_quality_gate(bundle):
    """TC06: Kiểm tra chất lượng mô hình (Quality Gate) trên tập test chuẩn cố định."""
    assert TEST_SPLIT_PATH.is_file(), f"Không tìm thấy tập test chuẩn tại {TEST_SPLIT_PATH}"

    test_df = pd.read_parquet(TEST_SPLIT_PATH)
    prepared, _ = prepare_model_frame(test_df, bundle["residualizer"], bundle["scaler"])

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for _, group in prepared.groupby("source_file", sort=False):
        values = group.loc[:, CNN_MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        label = int(group["fault_id"].iloc[0])
        if len(values) < WINDOW:
            continue
        for start in range(0, len(values) - WINDOW + 1, STRIDE):
            xs.append(values[start : start + WINDOW].T)
            ys.append(label)

    x_test = np.stack(xs)
    y_true = np.asarray(ys, dtype=np.int64)

    y_pred, _ = predict_windows(bundle["model"], x_test, bundle["device"])

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    assert acc >= MIN_ACCURACY, (
        f"Quality Gate thất bại: Accuracy đạt {acc:.4f}, thấp hơn ngưỡng tối thiểu {MIN_ACCURACY:.4f}"
    )
    assert macro_f1 >= MIN_MACRO_F1, (
        f"Quality Gate thất bại: Macro F1 đạt {macro_f1:.4f}, thấp hơn ngưỡng tối thiểu {MIN_MACRO_F1:.4f}"
    )

