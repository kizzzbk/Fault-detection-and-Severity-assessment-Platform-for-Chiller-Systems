"""Fit residualizer + scaler for the CA-TCC CNN (19 channels) and write sample.csv."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from chiller_fault_detection.config import DATA_DIR, DATASET_PARQUET, SHEET_NAME
from chiller_fault_detection.data.public.load import add_derived_features
from chiller_fault_detection.data.public.preprocess import (
    Residualizer,
    apply_residualizer,
    filter_kw_range,
    sort_within_files,
    time_splits,
    partition_by_split,
)
from demo.cnn_schema import (
    CKPT_NAME,
    CNN_FEATURE_COLUMNS,
    CNN_MODEL_FEATURE_COLUMNS,
    CNN_OPERATING_COLUMNS,
    CNN_RESIDUAL_COLUMNS,
    CNN_SOURCE_FEATURE_COLUMNS,
)
from demo.pipeline import DEMO_DIR, MODELS_DIR
from sklearn.linear_model import LinearRegression

EXTRAS = ["T_suc", "Tsh_dis", "TEA", "PO_feed"]
SAMPLE_PATH = DEMO_DIR / "sample.csv"
ARTIFACT_CKPT = REPO / "artifacts" / "ca_tcc_cnn" / CKPT_NAME


def _excel_extras(source_file: str) -> pd.DataFrame:
    path = DATA_DIR / source_file
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in ["Time", *EXTRAS] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} missing {missing}")
    out = raw.loc[:, ["Time", *EXTRAS]].copy()
    out = out.rename(columns={"Time": "time"})
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    for col in EXTRAS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time")


def _attach_extras(dataset: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for source, group in dataset.groupby("source_file", sort=False):
        extras = _excel_extras(str(source))
        block = group.sort_values("time").copy()
        merged = pd.merge_asof(
            block,
            extras,
            on="time",
            direction="nearest",
            tolerance=0.51,
        )
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    if "CDW_DT" not in out.columns or "CHW_DT" not in out.columns:
        out = add_derived_features(out)
    missing = [c for c in CNN_FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Reconstructed frame missing {missing}")
    out = out.dropna(subset=CNN_FEATURE_COLUMNS).reset_index(drop=True)
    return out


def _fit_residualizer(train: pd.DataFrame) -> Residualizer:
    normal = train.loc[train["fault"] == "normal"]
    model = LinearRegression()
    model.fit(normal.loc[:, CNN_OPERATING_COLUMNS], normal.loc[:, CNN_RESIDUAL_COLUMNS])
    return Residualizer(
        operating_columns=list(CNN_OPERATING_COLUMNS),
        residual_columns=list(CNN_RESIDUAL_COLUMNS),
        model=model,
        n_fit_rows=int(len(normal)),
    )


def _write_sample(frame: pd.DataFrame) -> None:
    plan = (
        ("normal", 800, "Benchmark Tests/normal b.xls", 800),
        ("cf", 400, "Condenser fouling/cf45.xls", 800),
        ("normal", 400, "Benchmark Tests/normal b.xls", 1600),
        ("eo", 400, "Excess oil/eo50.xls", 800),
    )
    parts = []
    for fault, n_rows, source, skip in plan:
        block = frame.loc[(frame["fault"] == fault) & (frame["source_file"] == source)]
        block = block.sort_values("time")
        slice_ = block.iloc[skip : skip + n_rows]
        if len(slice_) < n_rows:
            slice_ = block.iloc[-n_rows:]
        parts.append(slice_.copy())
    sample = pd.concat(parts, ignore_index=True)
    sample["time"] = np.arange(len(sample), dtype=np.float64) * 10.0
    keep = ["time", *CNN_SOURCE_FEATURE_COLUMNS]
    if "CDW_DT" in sample.columns:
        keep = ["time", *CNN_FEATURE_COLUMNS]
    sample.loc[:, [c for c in keep if c in sample.columns]].to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample → {SAMPLE_PATH} ({len(sample)} rows)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET_PARQUET)
    print(f"Attaching {EXTRAS} from Excel onto {len(dataset)} rows…")
    frame = _attach_extras(dataset)
    frame, n_drop = filter_kw_range(frame)
    print(f"After extras+kW: {len(frame)} rows (dropped {n_drop})")
    frame = sort_within_files(frame)
    frame = time_splits(frame)
    parts = partition_by_split(frame)
    residualizer = _fit_residualizer(parts["train"])
    scaled = {}
    scaler = StandardScaler()
    scaler.fit(apply_residualizer(parts["train"], residualizer)[CNN_MODEL_FEATURE_COLUMNS])
    for name, part in parts.items():
        residual = apply_residualizer(part, residualizer)
        residual[CNN_MODEL_FEATURE_COLUMNS] = scaler.transform(residual[CNN_MODEL_FEATURE_COLUMNS])
        scaled[name] = residual
    joblib.dump(residualizer, MODELS_DIR / "residualizer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    dest = MODELS_DIR / CKPT_NAME
    shutil.copy2(ARTIFACT_CKPT, dest)
    _write_sample(frame)
    print(
        f"Wrote {MODELS_DIR / 'residualizer.joblib'}, scaler, {dest} "
        f"(n_fit={residualizer.n_fit_rows}, channels={len(CNN_MODEL_FEATURE_COLUMNS)})"
    )


if __name__ == "__main__":
    main()
