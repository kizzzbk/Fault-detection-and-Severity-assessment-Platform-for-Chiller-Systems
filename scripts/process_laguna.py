"""Map a Laguna field chiller Excel export onto the processed ASHRAE schema."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chiller_fault_detection.config import PROCESSED_DIR, SHEET_NAME
from chiller_fault_detection.data.load import FEATURE_COLUMNS

DEFAULT_SOURCE = Path("/home/phananh/Downloads/LAGUNA CH01 892026.xls")
DEFAULT_SCHEMA = PROCESSED_DIR / "dataset.parquet"
DEFAULT_OUT = PROCESSED_DIR / "laguna_processed.parquet"

SOURCE_FILE_VALUE = "laguna.xls"
TIME_STEP_SECONDS = 10.0
TRAILING_ROWS_TO_DROP = 3

# ASHRAE column -> Laguna source column (matched case-insensitively).
DIRECT_MAP: dict[str, str] = {
    "TEI": "ECW",
    "TEO": "LCW",
    "TCI": "ECDW",
    "TCO": "LCDW",
    "TRE": "ERT",
    "TRC": "CRT",
    "T_suc": "EST",
    "Tsh_suc": "SUPRHEAT",
    "TR_dis": "CMPD",
    "TEA": "EVAP_APP",
    "TCA": "COND_APP",
    "kW": "kW",
    "TO_sump": "OILT",
    "PO_net": "OILPD",
    "PRE": "ERP",
    "PRC": "CRP",
}

# ASHRAE column -> (Laguna left operand, Laguna right operand, operator).
DERIVED_MAP: dict[str, tuple[str, str, str]] = {
    "PO_feed": ("OILPD", "ERP", "+"),
    "P_lift": ("CRP", "ERP", "-"),
    "Tsh_dis": ("CMPD", "CRT", "-"),
}

MAPPED_FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(DIRECT_MAP) + tuple(DERIVED_MAP) + ("CDW_DT", "CHW_DT")
)

SOURCE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *DIRECT_MAP.values(),
            *(left for left, _, _ in DERIVED_MAP.values()),
            *(right for _, right, _ in DERIVED_MAP.values()),
        ]
    )
)


def _normalize_name(name: object) -> str:
    return str(name).strip().upper().rstrip(":")


def _excel_engine(path: Path) -> str:
    """Pick xlrd vs openpyxl from magic bytes (.xls exports are often xlsx)."""
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic.startswith(b"PK"):
        return "openpyxl"
    return "xlrd"


def _find_schema_column(schema_columns: list[str], name: str) -> str:
    key = _normalize_name(name)
    for col in schema_columns:
        if _normalize_name(col) == key:
            return col
    raise KeyError(f"Schema is missing column {name!r}")


def read_schema_columns(schema_path: Path, sheet_name: str = SHEET_NAME) -> list[str]:
    suffix = schema_path.suffix.lower()
    if suffix == ".csv":
        header = pd.read_csv(schema_path, nrows=0)
        return [str(col).strip() for col in header.columns]
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return [str(col).strip() for col in pq.ParquetFile(schema_path).schema.names]
    header = pd.read_excel(
        schema_path,
        sheet_name=sheet_name,
        nrows=0,
        engine=_excel_engine(schema_path),
    )
    return [str(col).strip() for col in header.columns]


def _find_header_row(raw: pd.DataFrame, required: tuple[str, ...]) -> int:
    needed = {_normalize_name(name) for name in required}
    for idx, row in raw.iterrows():
        values = {_normalize_name(val) for val in row.tolist() if pd.notna(val)}
        if needed.issubset(values):
            return int(idx)
    raise ValueError(
        "Could not find a Laguna header row containing: " + ", ".join(required)
    )


def _column_lookup(columns: pd.Index) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in columns:
        key = _normalize_name(col)
        lookup.setdefault(key, str(col))
    return lookup


def _require_source_column(lookup: dict[str, str], name: str) -> str:
    key = _normalize_name(name)
    if key not in lookup:
        raise KeyError(f"Laguna file is missing source column {name!r}")
    return lookup[key]


def _numeric_series(frame: pd.DataFrame, lookup: dict[str, str], name: str) -> pd.Series:
    col = _require_source_column(lookup, name)
    return pd.to_numeric(frame[col], errors="coerce")


def read_laguna(source_path: Path) -> pd.DataFrame:
    engine = _excel_engine(source_path)
    xl = pd.ExcelFile(source_path, engine=engine)
    last_error: Exception | None = None
    for sheet_name in xl.sheet_names:
        raw = pd.read_excel(
            source_path, sheet_name=sheet_name, header=None, engine=engine
        )
        try:
            header_row = _find_header_row(raw, SOURCE_COLUMNS)
        except ValueError as exc:
            last_error = exc
            continue
        frame = raw.iloc[header_row + 1 :].copy()
        frame.columns = [str(col).strip() for col in raw.iloc[header_row].tolist()]
        return frame.reset_index(drop=True)
    raise ValueError(str(last_error) if last_error else "No usable Laguna sheet found")


def map_to_schema(source: pd.DataFrame, schema_columns: list[str]) -> pd.DataFrame:
    lookup = _column_lookup(source.columns)
    n_rows = len(source)
    out = pd.DataFrame(index=range(n_rows), columns=schema_columns)

    mapped: dict[str, pd.Series] = {}
    for dest, src_name in DIRECT_MAP.items():
        mapped[dest] = _numeric_series(source, lookup, src_name)

    for dest, (left, right, op) in DERIVED_MAP.items():
        left_s = _numeric_series(source, lookup, left)
        right_s = _numeric_series(source, lookup, right)
        if op == "+":
            mapped[dest] = left_s + right_s
        elif op == "-":
            mapped[dest] = left_s - right_s
        else:
            raise ValueError(f"Unsupported derived operator: {op}")

    # Same ΔT formulas as ASHRAE load (not Laguna's native CDW_DT / CHW_DT tags).
    mapped["CDW_DT"] = mapped["TCO"] - mapped["TCI"]
    mapped["CHW_DT"] = mapped["TEI"] - mapped["TEO"]

    keep = pd.concat(mapped, axis=1).notna().any(axis=1)
    for dest, series in mapped.items():
        if dest not in out.columns:
            raise KeyError(f"Schema is missing mapped column {dest!r}")
        out[dest] = series.astype("float64")

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            raise KeyError(f"Schema is missing FEATURE_COLUMNS entry {col!r}")
        if col not in mapped:
            out[col] = np.nan

    out = out.loc[keep].reset_index(drop=True)
    if len(out) <= TRAILING_ROWS_TO_DROP:
        raise ValueError(
            f"Only {len(out)} rows after mapping; cannot drop "
            f"{TRAILING_ROWS_TO_DROP} trailing rows"
        )
    out = out.iloc[:-TRAILING_ROWS_TO_DROP].reset_index(drop=True)

    time_col = _find_schema_column(schema_columns, "time")
    out[time_col] = np.arange(len(out), dtype="float64") * TIME_STEP_SECONDS

    source_col = _find_schema_column(schema_columns, "source_file")
    out[source_col] = SOURCE_FILE_VALUE

    return out.loc[:, schema_columns]


def process_laguna(
    source_path: Path,
    schema_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_columns = read_schema_columns(schema_path)
    source = read_laguna(source_path)
    frame = map_to_schema(source, schema_columns)

    os.makedirs(out_path.parent, exist_ok=True)
    frame.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    frame = process_laguna(args.source, args.schema, args.out)
    schema_columns = read_schema_columns(args.schema)
    missing_features = [
        col for col in FEATURE_COLUMNS if col not in MAPPED_FEATURE_COLUMNS
    ]
    populated = [
        col for col in MAPPED_FEATURE_COLUMNS if frame[col].notna().any()
    ]
    print(f"Wrote {len(frame)} rows × {len(frame.columns)} cols → {args.out}")
    print(
        f"schema_columns={len(schema_columns)} "
        f"order_match={list(frame.columns) == schema_columns}"
    )
    print(f"mapped_populated={len(populated)}/{len(MAPPED_FEATURE_COLUMNS)}: {populated}")
    print(f"feature_columns_nan={missing_features}")
    print(f"source_file={frame['source_file'].unique().tolist()}")
    print(f"time_range=[{frame['time'].iloc[0]}, {frame['time'].iloc[-1]}]")


if __name__ == "__main__":
    main()
