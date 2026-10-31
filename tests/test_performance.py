"""Tests for model classification accuracy and real-time inference latency."""
import time

import numpy as np
import pytest

from demo.cnn_schema import CNN_INPUT_CHANNELS
from demo.pipeline import (
    WINDOW,
    load_bundle,
    predict_windows,
)
from webapp.server import DATA

PARQUET_PATH = DATA / "twin_library.parquet"

@pytest.fixture(scope="module")
def bundle():
    return load_bundle()

def test_model_inference_latency(bundle):
    model = bundle["model"]
    device = bundle["device"]

    batch_size = 4
    dummy_input = np.random.randn(batch_size, CNN_INPUT_CHANNELS, WINDOW).astype(np.float32)
    predict_windows(model, dummy_input, device)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        predict_windows(model, dummy_input, device)
        latencies.append(time.perf_counter() - t0)
    mean_latency_ms = np.mean(latencies) * 1000.0

    assert mean_latency_ms < 200.0, f"Mô hình suy luận quá chậm: {mean_latency_ms:.2f}ms (Ngưỡng < 200ms)"