"""Load an uploaded series, match the test/val window path, predict fault_id."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from chiller_fault_detection.config import KW_MAX, KW_MIN
from chiller_fault_detection.data.public.load import add_derived_features
from chiller_fault_detection.data.public.preprocess import (
    Residualizer,
    apply_residualizer,
)
from chiller_fault_detection.labels import ID_TO_LABEL
from chiller_fault_detection.models.model import base_Model
from demo.cnn_schema import (
    CKPT_NAME,
    CNN_FEATURE_COLUMNS,
    CNN_INPUT_CHANNELS,
    CNN_MODEL_FEATURE_COLUMNS,
    CNN_OPERATING_COLUMNS,
    CNN_SOURCE_FEATURE_COLUMNS,
)

WINDOW = 128
STRIDE = 64
DEFAULT_DT_S = 10.0

LANE_KEYS: list[str] = [
    "ro",
    "rl",
    "nc",
    "fwe",
    "fwc",
    "eo",
    "cf",
    "normal",
]

LANE_LABELS: dict[str, str] = {
    "ro": "Refrigerant Overcharge (ro)",
    "rl": "Refrigerant Leak / Undercharge (rl)",
    "nc": "Non-condensable Gas (nc)",
    "fwe": "Reduced Evaporator Water Flow (fwe)",
    "fwc": "Reduced Condenser Water Flow (fwc)",
    "eo": "Excess Oil (eo)",
    "cf": "Condenser Fouling (cf)",
    "normal": "Normal",
}

SHORT_LABELS: dict[str, str] = {
    "ro": "Refrigerant Overcharge",
    "rl": "Refrigerant Leak / Undercharge",
    "nc": "Non-condensable Gas",
    "fwe": "Reduced Evaporator Water Flow",
    "fwc": "Reduced Condenser Water Flow",
    "eo": "Excess Oil",
    "cf": "Condenser Fouling",
    "normal": "Normal",
}

LANE_COLORS: dict[str, str] = {
    "normal": "#9CA3AF",
    "cf": "#E11D48",
    "eo": "#F59E0B",
    "fwc": "#2563EB",
    "fwe": "#059669",
    "nc": "#7C3AED",
    "rl": "#DB2777",
    "ro": "#0D9488",
}

DEMO_DIR = Path(__file__).resolve().parent
MODELS_DIR = DEMO_DIR / "models"
REPO_ROOT = DEMO_DIR.parent
CKPT_PATH = MODELS_DIR / CKPT_NAME
ARTIFACT_CKPT = REPO_ROOT / "artifacts" / "ca_tcc_cnn" / CKPT_NAME
SCALER_PATH = MODELS_DIR / "scaler.joblib"
RESIDUALIZER_PATH = MODELS_DIR / "residualizer.joblib"


def _device() -> str:
    return "cpu"


@dataclass
class WindowPred:
    start_idx: int
    end_idx: int
    t0_s: float
    t1_s: float
    duration_s: float
    fault: str
    fault_id: int
    confidence: float


@dataclass
class Segment:
    fault: str
    t0_s: float
    t1_s: float
    duration_s: float
    n_windows: int
    confidence: float


@dataclass
class Diagnosis:
    n_windows: int
    n_fault_types: int
    fault_seconds: float
    health_ratio: float
    by_fault_seconds: dict[str, float]
    windows: list[WindowPred]
    segments: list[Segment]
    t_span_s: float
    n_rows: int
    prepared_as: str


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours and minutes:
        return f"{hours}h {minutes}'"
    if hours:
        return f"{hours}h"
    if minutes:
        return f"{minutes}'"
    return f"{secs}s"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Upload a .csv or .parquet file.")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {c: c.strip() for c in out.columns}
    return out.rename(columns=rename)


def _ensure_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "time" not in out.columns:
        out["time"] = np.arange(len(out), dtype=np.float64) * DEFAULT_DT_S
        return out
    time = pd.to_numeric(out["time"], errors="coerce")
    if time.isna().all():
        parsed = pd.to_datetime(out["time"], errors="coerce")
        if parsed.notna().any():
            origin = parsed.dropna().iloc[0]
            time = (parsed - origin).dt.total_seconds()
    if time.isna().any():
        fill = np.arange(len(out), dtype=np.float64) * DEFAULT_DT_S
        time = time.fillna(pd.Series(fill, index=out.index))
    out["time"] = time.astype(np.float64)
    return out


def _infer_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return DEFAULT_DT_S
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return DEFAULT_DT_S
    return float(np.median(diffs))


def prepare_model_frame(frame: pd.DataFrame, residualizer: Residualizer, scaler) -> tuple[pd.DataFrame, str]:
    """Same path as train/val: residualize 17 channels, keep PO_net/PO_feed, scale 19."""
    frame = _normalize_columns(frame)
    frame = _ensure_time(frame)
    if "source_file" not in frame.columns:
        frame["source_file"] = "uploaded"

    has_knobs = all(col in frame.columns for col in CNN_OPERATING_COLUMNS)
    has_model = all(col in frame.columns for col in CNN_MODEL_FEATURE_COLUMNS)

    if has_knobs:
        missing = [c for c in CNN_SOURCE_FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"File is missing sensor columns the CNN was trained on: {', '.join(missing)}")
        if "CDW_DT" not in frame.columns or "CHW_DT" not in frame.columns:
            frame = add_derived_features(frame)
        for col in CNN_FEATURE_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "kW" in frame.columns:
            frame = frame.loc[frame["kW"].between(KW_MIN, KW_MAX, inclusive="both")].copy()
        frame = frame.dropna(subset=CNN_FEATURE_COLUMNS).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No rows left after the running-load filter and NaN drop.")
        prepared = apply_residualizer(frame, residualizer)
        prepared[CNN_MODEL_FEATURE_COLUMNS] = scaler.transform(prepared[CNN_MODEL_FEATURE_COLUMNS])
        return prepared.reset_index(drop=True), "raw+residual+scale"

    if has_model:
        for col in CNN_MODEL_FEATURE_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=CNN_MODEL_FEATURE_COLUMNS).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No rows left after dropping NaN in model columns.")
        return frame, "already-model-ready"

    raise ValueError(
        "File must contain the raw RP-1043 sensors the CA-TCC CNN was trained on "
        f"(including {', '.join(c for c in ('T_suc', 'Tsh_dis', 'TEA', 'PO_feed', 'TEO', 'TCI', 'kW'))})."
    )


def cut_windows(frame: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Same sliding window as ``window_part`` in trainer.py, plus time spans."""
    xs: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for _, group in frame.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        values = group.loc[:, CNN_MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        times = group["time"].to_numpy(dtype=np.float64)
        dt = _infer_dt(times)
        if len(values) < window:
            continue
        for start in range(0, len(values) - window + 1, stride):
            end = start + window
            vis_end = min(start + stride, len(values) - 1)
            if start + window >= len(values):
                vis_end = len(values) - 1
            t0 = float(times[start])
            t1 = float(times[vis_end])
            if t1 <= t0:
                t1 = t0 + stride * dt
            xs.append(values[start:end].T)
            meta.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "t0_s": t0,
                    "t1_s": t1,
                    "duration_s": t1 - t0,
                }
            )
    if not xs:
        raise ValueError(
            f"Need at least {window} rows after processing to cut a window "
            f"(got {len(frame)})."
        )
    return np.stack(xs), meta


def merge_segments(windows: list[WindowPred]) -> list[Segment]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.t0_s)
    groups: list[list[WindowPred]] = [[ordered[0]]]
    for item in ordered[1:]:
        prev = groups[-1][-1]
        if item.fault == prev.fault:
            groups[-1].append(item)
        else:
            groups.append([item])
    segments: list[Segment] = []
    for group in groups:
        segments.append(
            Segment(
                fault=group[0].fault,
                t0_s=group[0].t0_s,
                t1_s=group[-1].t1_s,
                duration_s=sum(w.duration_s for w in group),
                n_windows=len(group),
                confidence=float(np.mean([w.confidence for w in group])),
            )
        )
    return segments


def _resolve_ckpt(root: Path) -> Path:
    local = root / CKPT_NAME
    if local.is_file():
        return local
    if ARTIFACT_CKPT.is_file():
        return ARTIFACT_CKPT
    raise FileNotFoundError(
        f"CA-TCC CNN checkpoint not found at {local} or {ARTIFACT_CKPT}"
    )


def _cnn_features_len(length: int, kernel_size: int = 8, stride: int = 1) -> int:
    """Temporal length after the three conv blocks in ``base_Model``."""

    def _conv(size: int, kernel: int, step: int, padding: int) -> int:
        return (size + 2 * padding - kernel) // step + 1

    def _pool(size: int) -> int:
        return (size + 2 - 2) // 2 + 1

    length = _pool(_conv(length, kernel_size, stride, kernel_size // 2))
    length = _pool(_conv(length, 8, 1, 4))
    length = _pool(_conv(length, 8, 1, 4))
    return int(length)


def load_cnn_encoder(ckpt_path: Path, device: str):
    import torch

    # Keep this path free of trainer.py (matplotlib/seaborn/einops) so Cloud
    # can import the demo from demo/requirements.txt alone.
    config = SimpleNamespace(
        input_channels=CNN_INPUT_CHANNELS,
        kernel_size=8,
        stride=1,
        final_out_channels=128,
        dropout=0.35,
        features_len=_cnn_features_len(WINDOW),
        num_classes=len(ID_TO_LABEL),
    )
    model = base_Model(config)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_windows(model, windows: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """``windows`` is (N, C, T), same layout as ``window_part`` / ``model_evaluate``."""
    import torch
    import torch.nn.functional as F

    batches = []
    with torch.no_grad():
        for start in range(0, len(windows), 64):
            batch = torch.from_numpy(windows[start : start + 64]).float().to(device)
            logits, _ = model(batch)
            batches.append(F.softmax(logits, dim=1).cpu().numpy())
    proba = np.concatenate(batches, axis=0)
    return proba.argmax(axis=1), proba


def load_bundle(models_dir: Path | None = None) -> dict[str, Any]:
    root = models_dir or MODELS_DIR
    scaler_path = root / SCALER_PATH.name
    residual_path = root / RESIDUALIZER_PATH.name
    ckpt_path = _resolve_ckpt(root)
    if not scaler_path.is_file() or not residual_path.is_file():
        raise FileNotFoundError(
            "CNN residualizer/scaler missing. Run "
            "`uv run python scripts/fit_cnn_demo_preprocess.py` first."
        )
    device = _device()
    return {
        "model": load_cnn_encoder(ckpt_path, device),
        "device": device,
        "scaler": joblib.load(scaler_path),
        "residualizer": joblib.load(residual_path),
        "ckpt_path": ckpt_path,
    }


def diagnose_frame(frame: pd.DataFrame, artifacts: dict[str, Any] | None = None) -> Diagnosis:
    artifacts = artifacts or load_bundle()
    prepared, mode = prepare_model_frame(frame, artifacts["residualizer"], artifacts["scaler"])
    windows, meta = cut_windows(prepared)
    pred, proba = predict_windows(artifacts["model"], windows, artifacts["device"])

    preds: list[WindowPred] = []
    for i, row in enumerate(meta):
        fault_id = int(pred[i])
        fault = ID_TO_LABEL[fault_id].value
        preds.append(
            WindowPred(
                start_idx=int(row["start_idx"]),
                end_idx=int(row["end_idx"]),
                t0_s=float(row["t0_s"]),
                t1_s=float(row["t1_s"]),
                duration_s=float(row["duration_s"]),
                fault=fault,
                fault_id=fault_id,
                confidence=float(proba[i, fault_id]),
            )
        )

    by_fault = {key: 0.0 for key in LANE_KEYS}
    for item in preds:
        by_fault[item.fault] = by_fault.get(item.fault, 0.0) + item.duration_s
    total = sum(by_fault.values()) or 1.0
    fault_seconds = total - by_fault.get("normal", 0.0)
    n_fault_types = sum(1 for key, sec in by_fault.items() if key != "normal" and sec > 0)
    t0 = min(w.t0_s for w in preds)
    t1 = max(w.t1_s for w in preds)
    return Diagnosis(
        n_windows=len(preds),
        n_fault_types=n_fault_types,
        fault_seconds=fault_seconds,
        health_ratio=by_fault.get("normal", 0.0) / total,
        by_fault_seconds=by_fault,
        windows=preds,
        segments=merge_segments(preds),
        t_span_s=t1 - t0,
        n_rows=len(prepared),
        prepared_as=mode,
    )


def diagnose_path(path: Path, artifacts: dict[str, Any] | None = None) -> Diagnosis:
    return diagnose_frame(_read_table(path), artifacts)
