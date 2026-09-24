"""Preprocess loaded RP-1043 parquet for training.

Pipeline
--------
1. Load ``dataset.parquet`` (light schema check only; full validation is in load).
2. Sort rows by ``time`` within each ``source_file``.
3. Assign train/val/test by **time order inside each file** (default 70/15/15).
4. Partition into three DataFrames; split X / y via ``load.split_xy``.
5. Fit ``StandardScaler`` on train only; transform val/test.
6. Save split artifacts under ``artifacts/processed/splits/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from chiller_fault_detection.config import (
    DATASET_PARQUET,
    SCALER_PATH,
    SPLIT_INFO_JSON,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from chiller_fault_detection.data.load import FEATURE_COLUMNS, split_xy

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
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    fault_counts_train: dict[str, int] = field(default_factory=dict)
    fault_counts_val: dict[str, int] = field(default_factory=dict)
    fault_counts_test: dict[str, int] = field(default_factory=dict)
    path_label_counts_train: dict[str, int] = field(default_factory=dict)
    path_label_counts_val: dict[str, int] = field(default_factory=dict)
    path_label_counts_test: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_processed_dataset(path: Path | None = None) -> pd.DataFrame:
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


def assign_time_splits(
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
        # Ensure each split gets at least one row when n >= 3.
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val

        positions = list(idx)
        train_idx = positions[:n_train]
        val_idx = positions[n_train : n_train + n_val]
        test_idx = positions[n_train + n_val :]
        assert len(test_idx) == n_test

        split_labels.loc[train_idx] = "train"
        split_labels.loc[val_idx] = "val"
        split_labels.loc[test_idx] = "test"

    out[SPLIT_COLUMN] = split_labels.to_numpy()
    return out


def partition_by_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame that already has a ``split`` column into three frames."""
    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Expected column {SPLIT_COLUMN!r} from assign_time_splits.")

    parts: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        part = df.loc[df[SPLIT_COLUMN] == name].drop(columns=[SPLIT_COLUMN])
        parts[name] = part.reset_index(drop=True)
        if parts[name].empty:
            raise ValueError(f"Split {name!r} is empty after time partitioning.")
    return parts


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


def fit_transform_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train only; transform all three splits."""
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(X_train[FEATURE_COLUMNS])
    val_arr = scaler.transform(X_val[FEATURE_COLUMNS])
    test_arr = scaler.transform(X_test[FEATURE_COLUMNS])

    X_train_s = pd.DataFrame(train_arr, columns=FEATURE_COLUMNS, index=X_train.index)
    X_val_s = pd.DataFrame(val_arr, columns=FEATURE_COLUMNS, index=X_val.index)
    X_test_s = pd.DataFrame(test_arr, columns=FEATURE_COLUMNS, index=X_test.index)
    return X_train_s, X_val_s, X_test_s, scaler


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
    df = load_processed_dataset(dataset_path)
    df = sort_within_files(df)
    df = assign_time_splits(
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

    xy = make_xy_splits(parts, target="both")
    X_train_s, X_val_s, X_test_s, scaler = fit_transform_scaler(
        xy["X_train"], xy["X_val"], xy["X_test"]
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
        report=report,
        out_dir=out_dir,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    args = parser.parse_args()

    report = run_preprocess(
        args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "n_rows_train": report.n_rows_train,
                "n_rows_val": report.n_rows_val,
                "n_rows_test": report.n_rows_test,
                "n_files": report.n_files,
                "fault_counts_train": report.fault_counts_train,
                "path_label_counts_test": report.path_label_counts_test,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
