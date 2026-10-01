"""Load ASHRAE RP-1043 Excel tests into one parquet table.

Policies (confirmed)
--------------------
* Sheet: ``Complete Data Set``
* Features: 22 fixed columns in ``FEATURE_COLUMNS``
* NaN in any feature → **drop** the row; count in ``LoadReport``
* Eligible file missing a required column → **raise** with full report
* Exclusion list is **exact** filenames in ``EXCLUDED_FILES``
* Metadata columns: source_file, time, fault, fault_id, severity_level, path_label
  (no ``severity_pct`` in the saved dataset)
* Tiny values ``abs(x) < 1e-20`` → ``0.0``; count printed / reported
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from chiller_fault_detection.config import (
    DATA_DIR,
    DATASET_META_JSON,
    DATASET_PARQUET,
    SHEET_NAME,
    TIME_COLUMN,
    TINY_ABS,
)
from chiller_fault_detection.labels import (
    VALID_FAULT_IDS,
    VALID_SEVERITY_LEVELS,
    SourceLabel,
    parse_source_label,
)

FEATURE_COLUMNS: list[str] = [
    "TEI",
    "TEO",
    "TCI",
    "TCO",
    "TRE",
    "TRC",
    "TRC_sub",
    "T_suc",
    "Tsh_suc",
    "TR_dis",
    "Tsh_dis",
    "TEA",
    "TCA",
    "kW",
    "TO_sump",
    "PO_net",
    "PRE",
    "PRC",
    "P_lift",
    "PO_feed",
    "FWC",
    "FWE",
]

# Group 1: experimental knobs. Used only as inputs to residual f_c.
OPERATING_COLUMNS: list[str] = ["TEO", "TCI", "kW"]

# Groups 2+3: residualize against OPERATING_COLUMNS (fit on train ∩ normal).
RESIDUAL_COLUMNS: list[str] = [
    "TEI",
    "TCO",
    "TRE",
    "T_suc",
    "Tsh_suc",
    "Tsh_dis",
    "PRE",
    "TRC",
    "TRC_sub",
    "TR_dis",
    "TEA",
    "TCA",
    "PRC",
    "P_lift",
    "TO_sump",
]

# Group 4: keep raw; not residualized.
KEEP_RAW_COLUMNS: list[str] = ["FWE", "FWC", "PO_net", "PO_feed"]

# Columns written into model-ready X after residual + drop knobs (19).
MODEL_FEATURE_COLUMNS: list[str] = [*RESIDUAL_COLUMNS, *KEEP_RAW_COLUMNS]

METADATA_COLUMNS: list[str] = [
    "source_file",
    "time",
    "fault",
    "fault_id",
    "severity_level",
    "path_label",
]

# Exact basename → reason. Matched case-insensitively.
EXCLUDED_FILES: dict[str, str] = {
    "read me first.xls": "readme",
    "11000.xlsx": "prebuilt_dataset",
    "rl40 alt--many unsteady tests.xls": "unsteady_excluded",
    "fwe40 alt--unsteady test8.xls": "unsteady_excluded",
    "fwe20 alt--unsteady test5.xls": "unsteady_excluded",
    "nc trace.xls": "nc_trace_excluded",
    "nc trace2.xls": "nc_trace_excluded",
    "modified nc.xls": "modified_nc_excluded",
    "aborted nc.xls": "aborted_excluded",
    "fwe20fwc20.xls": "multiple_faults",
    "aborted eo86.xls": "aborted_excluded",
    "eo73 alt--unsteady test1.xls": "unsteady_excluded",
    "defective pilot valve.xls": "dpv_not_in_taxonomy",
    "cf6.xls": "cf6_excluded",
    "near normal1.xls": "near_normal",
    "near normal2.xls": "near_normal",
    "near normal3.xls": "near_normal",
}


@dataclass
class LoadReport:
    discovered_files: int = 0
    loaded_files: int = 0
    loaded_rows: int = 0
    dropped_nan_rows: int = 0
    replaced_tiny_values: int = 0
    skipped_files: list[dict[str, str]] = field(default_factory=list)
    fault_counts: dict[str, int] = field(default_factory=dict)
    severity_level_counts: dict[str, int] = field(default_factory=dict)
    path_label_counts: dict[str, int] = field(default_factory=dict)
    sheet: str = SHEET_NAME
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    nan_policy: str = "drop rows with any NaN in FEATURE_COLUMNS"
    tiny_policy: str = f"abs(x) < {TINY_ABS} -> 0.0"
    missing_column_policy: str = "raise if any eligible file is missing FEATURE_COLUMNS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def default_output_path() -> Path:
    """Absolute parquet path from ``config.DATASET_PARQUET``."""
    return DATASET_PARQUET


def discover_excel_files(data_dir: Path | None = None) -> list[Path]:
    data_dir = data_dir or DATA_DIR
    files = sorted(data_dir.rglob("*.xls")) + sorted(data_dir.rglob("*.xlsx"))
    return files


def get_exclusion_reason(path: Path) -> str | None:
    """Return skip reason if ``path.name`` is in ``EXCLUDED_FILES``, else None."""
    return EXCLUDED_FILES.get(path.name.lower())


def inspect_columns(
    files: list[Path],
    *,
    sheet_name: str = SHEET_NAME,
) -> list[dict[str, Any]]:
    """Check each file for required feature columns on ``sheet_name``."""
    report: list[dict[str, Any]] = []
    for path in files:
        row: dict[str, Any] = {
            "file": str(path),
            "ok": False,
            "missing": [],
            "error": None,
        }
        try:
            xl = pd.ExcelFile(path)
            if sheet_name not in xl.sheet_names:
                row["error"] = (
                    f"missing sheet {sheet_name!r}; has {xl.sheet_names}"
                )
                report.append(row)
                continue
            header = pd.read_excel(
                path, sheet_name=sheet_name, nrows=0, engine=_excel_engine(path)
            )
            present = {str(c).strip() for c in header.columns}
            required = [TIME_COLUMN, *FEATURE_COLUMNS]
            missing = [c for c in required if c not in present]
            row["missing"] = missing
            row["ok"] = not missing
        except Exception as exc:  # noqa: BLE001 — collect per-file failures
            row["error"] = f"{type(exc).__name__}: {exc}"
        report.append(row)
    return report


def _excel_engine(path: Path) -> str:
    if path.suffix.lower() == ".xlsx":
        return "openpyxl"
    return "xlrd"


def read_excel_file(
    path: Path,
    *,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Read one Excel sheet. Assumes ``inspect_columns`` already passed."""
    df = pd.read_excel(path, sheet_name=sheet_name, engine=_excel_engine(path))
    df.columns = [str(c).strip() for c in df.columns]
    out = df.loc[:, [TIME_COLUMN, *FEATURE_COLUMNS]].copy()
    return out.rename(columns={TIME_COLUMN: "time"})


def coerce_feature_types(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cast features to float64; zero absolute values below ``TINY_ABS``."""
    out = df.copy()
    n_tiny = 0
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        tiny = out[col].notna() & (out[col].abs() < TINY_ABS) & (out[col] != 0)
        n = int(tiny.sum())
        if n:
            print(f"  {col}: replaced {n} tiny value(s) (|x| < {TINY_ABS}) with 0.0")
            out.loc[tiny, col] = 0.0
            n_tiny += n
        out[col] = out[col].astype("float64")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    return out, n_tiny


def drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.dropna(subset=FEATURE_COLUMNS).copy()
    return cleaned, before - len(cleaned)


def attach_metadata(
    df: pd.DataFrame,
    path: Path,
    data_dir: Path,
    label: SourceLabel,
) -> pd.DataFrame:
    out = df.copy()
    out["source_file"] = str(path.relative_to(data_dir))
    out["fault"] = label.fault.value
    out["fault_id"] = label.fault_id
    out["severity_level"] = label.severity_level
    out["path_label"] = label.path_label
    return out.loc[:, METADATA_COLUMNS + FEATURE_COLUMNS]


def validate_dataset(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("Dataset has 0 rows after loading.")

    missing_cols = [
        c for c in METADATA_COLUMNS + FEATURE_COLUMNS if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Dataset still contains NaN in FEATURE_COLUMNS.")

    bad_fault = set(df["fault_id"].unique()) - VALID_FAULT_IDS
    if bad_fault:
        raise ValueError(f"Invalid fault_id values: {sorted(bad_fault)}")

    bad_sl = set(df["severity_level"].unique()) - VALID_SEVERITY_LEVELS
    if bad_sl:
        raise ValueError(f"Invalid severity_level values: {sorted(bad_sl)}")

    normal_bad = df.loc[df["fault"] == "normal", "severity_level"]
    if len(normal_bad) and not normal_bad.eq(0).all():
        raise ValueError("Normal rows must have severity_level == 0")

    faulted = df.loc[df["fault"] != "normal", "severity_level"]
    if len(faulted) and not faulted.between(1, 4).all():
        raise ValueError("Faulted rows must have severity_level in 1..4")


def load_dataset(
    data_dir: Path | None,
    *,
    sheet_name: str = SHEET_NAME,
) -> tuple[pd.DataFrame, LoadReport]:
    data_dir = data_dir or DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    report = LoadReport(sheet=sheet_name)
    all_files = discover_excel_files(data_dir)
    report.discovered_files = len(all_files)

    eligible: list[Path] = []
    for path in all_files:
        reason = get_exclusion_reason(path)
        if reason is not None:
            report.skipped_files.append(
                {
                    "file": str(path.relative_to(data_dir)),
                    "reason": reason,
                }
            )
            continue
        eligible.append(path)

    column_report = inspect_columns(eligible, sheet_name=sheet_name)
    failures = [
        {
            "file": str(Path(r["file"]).relative_to(data_dir)),
            "missing": r["missing"],
            "error": r["error"],
        }
        for r in column_report
        if not r["ok"]
    ]
    if failures:
        raise ValueError(
            "Eligible files are missing required columns or failed to read:\n"
            + json.dumps(failures, indent=2)
        )

    frames: list[pd.DataFrame] = []
    for path in eligible:
        print(f"Loading {path.relative_to(data_dir)} ...")
        label = parse_source_label(path)
        part = read_excel_file(path, sheet_name=sheet_name)
        part, n_tiny = coerce_feature_types(part)
        report.replaced_tiny_values += n_tiny
        part, n_drop = drop_invalid_rows(part)
        report.dropped_nan_rows += n_drop
        if part.empty:
            report.skipped_files.append(
                {
                    "file": str(path.relative_to(data_dir)),
                    "reason": "all_rows_nan",
                }
            )
            continue
        part = attach_metadata(part, path, data_dir, label)
        frames.append(part)
        report.loaded_files += 1

    if not frames:
        raise ValueError("No rows loaded (all files skipped or empty).")

    df = pd.concat(frames, ignore_index=True)
    df["fault_id"] = df["fault_id"].astype("int64")
    df["severity_level"] = df["severity_level"].astype("int64")
    validate_dataset(df)

    report.loaded_rows = int(len(df))
    report.fault_counts = {
        str(k): int(v) for k, v in df["fault"].value_counts().sort_index().items()
    }
    report.severity_level_counts = {
        str(k): int(v)
        for k, v in df["severity_level"].value_counts().sort_index().items()
    }
    report.path_label_counts = {
        str(k): int(v)
        for k, v in df["path_label"].value_counts().sort_index().items()
    }
    return df, report


def save_parquet(
    df: pd.DataFrame,
    report: LoadReport | None = None,
    path: Path | None = None,
) -> Path:
    """Write ``dataset.parquet`` (+ ``.meta.json``).

    Requires ``pyarrow`` (declared in ``pyproject.toml``). If it is not
    installed, falls back to a sibling ``.csv`` so local work can continue,
    and still writes the metadata JSON.
    """
    path = path or default_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if report is not None:
        meta_path = (
            DATASET_META_JSON
            if path.resolve() == default_output_path().resolve()
            else path.with_suffix(".meta.json")
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"Wrote metadata → {meta_path}")

    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(
            f"WARNING: parquet engine missing ({exc}). "
            f"Wrote CSV fallback → {csv_path}. "
            "Install with: uv add pyarrow"
        )
        return csv_path
    return path
