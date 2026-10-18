"""Preprocess loaded RP-1043 parquet for training.

Pipeline
--------
1. Load ``dataset.parquet`` (light schema check only; full validation is in load).
2. Sort rows by ``time`` within each ``source_file``.
3. Assign train/val/test by **time order inside each file** (default 70/15/15).
4. Partition into three DataFrames. ``X`` keeps metadata and features; ``y`` holds labels.
5. Residualize groups 2+3 (incl. ``CDW_DT``, ``CHW_DT``): fit linear
   ``f_c(TEO, TCI, kW)`` on train∩normal, subtract from train/val/test;
   drop the three knobs; keep group 4 (``PO_net``, ``PO_feed``) raw.
6. Fit ``StandardScaler`` on the 19 model columns (train only).
7. Save split artifacts under ``artifacts/processed/splits/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from chiller_fault_detection.config import (
    DATASET_PARQUET,
    KW_MAX,
    KW_MIN,
    RESIDUALIZER_PATH,
    SCALER_PATH,
    SPLIT_INFO_JSON,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from chiller_fault_detection.data.public.load import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OPERATING_COLUMNS,
    RESIDUAL_COLUMNS,
)

REQUIRED_COLUMNS: list[str] = [
    "source_file",
    "time",
    "fault",
    "fault_id",
    "severity_level",
    "path_label",
    *FEATURE_COLUMNS,
]

SPLIT_COLUMN = "split"

@dataclass
class PreprocessReport:
    n_rows_total: int = 0
    n_rows_train: int = 0
    n_rows_val: int = 0
    n_rows_test: int = 0
    n_files: int = 0
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    split_method: str = "time_within_file"
    scaler_type: str = "StandardScaler"
    n_rows_dropped_kw: int = 0
    n_residual_fit_rows: int = 0
    feature_columns: list[str] = field(default_factory=lambda: list(MODEL_FEATURE_COLUMNS))
    fault_counts_train: dict[str, int] = field(default_factory=dict)
    fault_counts_val: dict[str, int] = field(default_factory=dict)
    fault_counts_test: dict[str, int] = field(default_factory=dict)
    path_label_counts_train: dict[str, int] = field(default_factory=dict)
    path_label_counts_val: dict[str, int] = field(default_factory=dict)
    path_label_counts_test: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read parquet with a light schema check (not full load.validate_dataset)."""
    path = path or DATASET_PARQUET
    if not path.is_file():
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.is_file():
            path = csv_fallback
        else:
            raise FileNotFoundError(
                f"Processed dataset not found: {path} (or {csv_fallback})"
            )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Processed dataset is empty.")
    return df


def sort_within_files(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each ``source_file`` block by ascending ``time``."""
    out = df.sort_values(
        ["source_file", "time"],
        kind="mergesort",
        ignore_index=True,
    )
    return out


def time_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> pd.DataFrame:
    """Label each row ``train`` / ``val`` / ``test`` by time order within file.

    For every ``source_file`` (already time-sorted):
    - first ``train_ratio`` fraction of rows → train
    - next ``val_ratio`` → val
    - remainder → test  (absorbs rounding so no rows are dropped)
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}={total}"
        )

    out = df.copy()
    split_labels = pd.Series(index=out.index, dtype="object")

    for _source, idx in out.groupby("source_file", sort=False).groups.items():
        n = len(idx)
        if n < 3:
            raise ValueError(
                f"File has only {n} rows; need at least 3 for time split: {_source}"
            )
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = n - n_train - n_val

        positions = list(idx)
        train_idx = positions[:n_train]
        val_idx = positions[n_train : n_train + n_val]
        test_idx = positions[n_train + n_val :]

        split_labels.loc[train_idx] = "train"
        split_labels.loc[val_idx] = "val"
        split_labels.loc[test_idx] = "test"

    out[SPLIT_COLUMN] = split_labels.to_numpy()
    return out


def partition_by_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame that already has a ``split`` column into three frames."""
    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Expected column {SPLIT_COLUMN!r} from time_splits.")

    parts: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        part = df.loc[df[SPLIT_COLUMN] == name].drop(columns=[SPLIT_COLUMN])
        parts[name] = part.reset_index(drop=True)
        if parts[name].empty:
            raise ValueError(f"Split {name!r} is empty after time partitioning.")
    return parts


def split_xy(
    df: pd.DataFrame,
    target: Literal["fault", "severity", "both"] = "both",
) -> (
    tuple[pd.DataFrame, pd.Series]
    | tuple[pd.DataFrame, pd.Series, pd.Series]
):
    X = df.copy()
    if target == "fault":
        return X, df["fault_id"].copy()
    if target == "severity":
        return X, df["severity_level"].copy()
    if target == "both":
        return X, df["fault_id"].copy(), df["severity_level"].copy()
    raise ValueError("target must be 'fault', 'severity', or 'both'")


def make_xy_splits(
    parts: dict[str, pd.DataFrame],
    *,
    target: str = "both",
) -> dict[str, Any]:
    """Apply ``split_xy`` to train/val/test partitions."""
    result: dict[str, Any] = {}
    for name, part in parts.items():
        xy = split_xy(part, target=target)  # type: ignore[arg-type]
        if target == "both":
            X, y_fault, y_severity = xy
            result[f"X_{name}"] = X
            result[f"y_fault_{name}"] = y_fault
            result[f"y_severity_{name}"] = y_severity
        elif target == "fault":
            X, y_fault = xy
            result[f"X_{name}"] = X
            result[f"y_fault_{name}"] = y_fault
        else:
            X, y_severity = xy
            result[f"X_{name}"] = X
            result[f"y_severity_{name}"] = y_severity
    return result


def filter_kw_range(
    df: pd.DataFrame,
    *,
    kw_min: float = KW_MIN,
    kw_max: float = KW_MAX,
) -> tuple[pd.DataFrame, int]:
    """Keep rows with physically plausible running load. Before the time split."""
    if "kW" not in df.columns:
        raise ValueError("Expected column 'kW' for the load filter.")
    mask = df["kW"].between(kw_min, kw_max, inclusive="both")
    n_drop = int((~mask).sum())
    return df.loc[mask].copy().reset_index(drop=True), n_drop


@dataclass
class Residualizer:
    """Linear f_c(TEO, TCI, kW) for each residual column; sklearn multi-output OLS."""

    operating_columns: list[str]
    residual_columns: list[str]
    model: LinearRegression
    n_fit_rows: int = 0


def fit_residualizer(X_train: pd.DataFrame) -> Residualizer:
    """Fit on train rows with fault == normal only."""
    needed = ["fault", *OPERATING_COLUMNS, *RESIDUAL_COLUMNS]
    missing = [c for c in needed if c not in X_train.columns]
    if missing:
        raise ValueError(f"X_train missing columns for residual fit: {missing}")
    normal = X_train.loc[X_train["fault"] == "normal"]
    if len(normal) == 0:
        raise ValueError("No normal rows in X_train to fit residualizer.")
    model = LinearRegression()
    model.fit(normal.loc[:, OPERATING_COLUMNS], normal.loc[:, RESIDUAL_COLUMNS])
    return Residualizer(
        operating_columns=list(OPERATING_COLUMNS),
        residual_columns=list(RESIDUAL_COLUMNS),
        model=model,
        n_fit_rows=int(len(normal)),
    )


def apply_residualizer(X: pd.DataFrame, residualizer: Residualizer) -> pd.DataFrame:
    """Overwrite residual columns with x - f(TEO, TCI, kW); drop the knobs."""
    missing = [
        c
        for c in residualizer.operating_columns + residualizer.residual_columns
        if c not in X.columns
    ]
    if missing:
        raise ValueError(f"Frame missing columns for residual transform: {missing}")
    out = X.copy()
    pred = residualizer.model.predict(out.loc[:, residualizer.operating_columns])
    out.loc[:, residualizer.residual_columns] = (
        out.loc[:, residualizer.residual_columns].to_numpy() - pred
    )
    return out.drop(columns=list(residualizer.operating_columns))


def fit_transform_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train model columns; keep every other column."""
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(X_train[MODEL_FEATURE_COLUMNS])
    val_arr = scaler.transform(X_val[MODEL_FEATURE_COLUMNS])
    test_arr = scaler.transform(X_test[MODEL_FEATURE_COLUMNS])

    def _with_scaled(frame: pd.DataFrame, values) -> pd.DataFrame:
        out = frame.copy()
        out[MODEL_FEATURE_COLUMNS] = values
        return out

    return (
        _with_scaled(X_train, train_arr),
        _with_scaled(X_val, val_arr),
        _with_scaled(X_test, test_arr),
        scaler,
    )


def build_preprocess_report(
    parts: dict[str, pd.DataFrame],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> PreprocessReport:
    def _counts(frame: pd.DataFrame, col: str) -> dict[str, int]:
        return {
            str(k): int(v) for k, v in frame[col].value_counts().sort_index().items()
        }

    n_files = int(
        pd.concat(parts.values(), ignore_index=True)["source_file"].nunique()
    )
    return PreprocessReport(
        n_rows_total=sum(len(p) for p in parts.values()),
        n_rows_train=len(parts["train"]),
        n_rows_val=len(parts["val"]),
        n_rows_test=len(parts["test"]),
        n_files=n_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fault_counts_train=_counts(parts["train"], "fault"),
        fault_counts_val=_counts(parts["val"], "fault"),
        fault_counts_test=_counts(parts["test"], "fault"),
        path_label_counts_train=_counts(parts["train"], "path_label"),
        path_label_counts_val=_counts(parts["val"], "path_label"),
        path_label_counts_test=_counts(parts["test"], "path_label"),
    )


def save_preprocess_artifacts(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_fault_train: pd.Series,
    y_fault_val: pd.Series,
    y_fault_test: pd.Series,
    y_severity_train: pd.Series,
    y_severity_val: pd.Series,
    y_severity_test: pd.Series,
    scaler: StandardScaler,
    residualizer: Residualizer | None = None,
    report: PreprocessReport,
    out_dir: Path | None = None,
) -> Path:
    """Write scaled X, labels, scaler, and split_info.json."""
    out_dir = out_dir or SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_frame(name: str, frame: pd.DataFrame) -> None:
        path = out_dir / f"{name}.parquet"
        try:
            frame.to_parquet(path, index=False)
        except ImportError:
            frame.to_csv(path.with_suffix(".csv"), index=False)

    def _save_series(name: str, series: pd.Series) -> None:
        frame = series.to_frame(name=series.name or name)
        _save_frame(name, frame)

    _save_frame("X_train", X_train)
    _save_frame("X_val", X_val)
    _save_frame("X_test", X_test)
    _save_series("y_fault_train", y_fault_train.rename("fault_id"))
    _save_series("y_fault_val", y_fault_val.rename("fault_id"))
    _save_series("y_fault_test", y_fault_test.rename("fault_id"))
    _save_series("y_severity_train", y_severity_train.rename("severity_level"))
    _save_series("y_severity_val", y_severity_val.rename("severity_level"))
    _save_series("y_severity_test", y_severity_test.rename("severity_level"))

    scaler_path = out_dir / SCALER_PATH.name
    joblib.dump(scaler, scaler_path)
    if residualizer is not None:
        residual_path = out_dir / RESIDUALIZER_PATH.name
        joblib.dump(residualizer, residual_path)
        print(f"Wrote residualizer → {residual_path}")

    info_path = out_dir / SPLIT_INFO_JSON.name
    info_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"Wrote splits → {out_dir}")
    print(f"Wrote scaler → {scaler_path}")
    print(f"Wrote report → {info_path}")
    return out_dir


def run_preprocess(
    dataset_path: Path | None = None,
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    out_dir: Path | None = None,
) -> PreprocessReport:
    """Full preprocess orchestration."""
    df = load_dataset(dataset_path)
    df, n_dropped_kw = filter_kw_range(df)
    df = sort_within_files(df)
    df = time_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    parts = partition_by_split(df)
    report = build_preprocess_report(
        parts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    report.n_rows_dropped_kw = n_dropped_kw

    xy = make_xy_splits(parts, target="both")
    residualizer = fit_residualizer(xy["X_train"])
    report.n_residual_fit_rows = residualizer.n_fit_rows
    X_train_r = apply_residualizer(xy["X_train"], residualizer)
    X_val_r = apply_residualizer(xy["X_val"], residualizer)
    X_test_r = apply_residualizer(xy["X_test"], residualizer)
    X_train_s, X_val_s, X_test_s, scaler = fit_transform_scaler(
        X_train_r, X_val_r, X_test_r
    )
    save_preprocess_artifacts(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_fault_train=xy["y_fault_train"],
        y_fault_val=xy["y_fault_val"],
        y_fault_test=xy["y_fault_test"],
        y_severity_train=xy["y_severity_train"],
        y_severity_val=xy["y_severity_val"],
        y_severity_test=xy["y_severity_test"],
        scaler=scaler,
        residualizer=residualizer,
        report=report,
        out_dir=out_dir,
    )
    return report