import numpy as np
import pandas as pd
import pytest

from demo.cnn_schema import (
    CNN_FEATURE_COLUMNS,
    CNN_INPUT_CHANNELS,
    CNN_MODEL_FEATURE_COLUMNS,
)
from demo.pipeline import (
    WINDOW,
    cut_windows,
    diagnose_frame,
    load_bundle,
    predict_windows,
)


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
