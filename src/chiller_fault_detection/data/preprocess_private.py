"""Map a private Laguna field export onto the processed ASHRAE schema.

Steps in this module
--------------------
1. ``PrivateColumnMap`` — ASHRAE column contract (direct, derived, ΔT).
2. ``ExcelEngine`` — pick xlrd or openpyxl from file magic bytes.
3. ``SchemaReader`` — read the target column order (csv, parquet, or Excel).
4. ``LagunaReader`` — find the header row and return the raw table.
5. ``UnitConverter`` — °C→°F for absolute and difference temperatures,
   kPa→psi for pressures, on the raw Laguna columns.
6. ``SchemaMapper`` — map columns, derive values, drop empty and trailing
   rows, stamp ``time`` and ``source_file``.
7. ``PrivatePreprocessor`` — run the steps above and write parquet.
"""

from __future__ import annotations

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

C_TO_F_SCALE = 1.8
C_TO_F_OFFSET = 32.0
KPA_PER_PSI = 6.89476


def _normalize_name(name: object) -> str:
    return str(name).strip().upper().rstrip(":")


class PrivateColumnMap:
    """ASHRAE destination columns and the Laguna tags they come from."""

    # ASHRAE column -> Laguna source column (matched case-insensitively).
    DIRECT: dict[str, str] = {
        "TEI": "ECW",
        "TEO": "LCW",
        "TCI": "ECDW",
        "TCO": "LCDW",
        "TRE": "ERT",
        "TRC": "CRT",
        "Tsh_suc": "SUPRHEAT",
        "TR_dis": "CMPD",
        "TCA": "COND_APP",
        "kW": "kW",
        "TO_sump": "OILT",
        "PO_net": "OILPD",
        "PRE": "ERP",
        "PRC": "CRP",
    }

    # ASHRAE column -> (Laguna left operand, Laguna right operand, operator).
    DERIVED: dict[str, tuple[str, str, str]] = {
        "P_lift": ("CRP", "ERP", "-"),
    }

    DELTA_T: tuple[str, ...] = ("CDW_DT", "CHW_DT")

    @classmethod
    def mapped_feature_columns(cls) -> tuple[str, ...]:
        return tuple(cls.DIRECT) + tuple(cls.DERIVED) + cls.DELTA_T

    @classmethod
    def source_columns(cls) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *cls.DIRECT.values(),
                    *(left for left, _, _ in cls.DERIVED.values()),
                    *(right for _, right, _ in cls.DERIVED.values()),
                ]
            )
        )

    @staticmethod
    def apply_operator(left: pd.Series, right: pd.Series, op: str) -> pd.Series:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        raise ValueError(f"Unsupported derived operator: {op}")


class ExcelEngine:
    """Pick xlrd vs openpyxl. Laguna ``.xls`` exports are often xlsx bytes."""

    @staticmethod
    def for_path(path: Path) -> str:
        with path.open("rb") as handle:
            magic = handle.read(8)
        if magic.startswith(b"PK"):
            return "openpyxl"
        return "xlrd"


class SchemaReader:
    """Target column order taken from an already-processed ASHRAE file."""

    def __init__(self, path: Path, sheet_name: str = SHEET_NAME) -> None:
        self.path = path
        self.sheet_name = sheet_name

    def columns(self) -> list[str]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            header = pd.read_csv(self.path, nrows=0)
            return [str(col).strip() for col in header.columns]
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            return [str(col).strip() for col in pq.ParquetFile(self.path).schema.names]
        header = pd.read_excel(
            self.path,
            sheet_name=self.sheet_name,
            nrows=0,
            engine=ExcelEngine.for_path(self.path),
        )
        return [str(col).strip() for col in header.columns]

    @staticmethod
    def find(columns: list[str], name: str) -> str:
        key = _normalize_name(name)
        for col in columns:
            if _normalize_name(col) == key:
                return col
        raise KeyError(f"Schema is missing column {name!r}")


class LagunaReader:
    """Read the Laguna workbook and return the table under its header row."""

    def __init__(
        self,
        path: Path,
        column_map: type[PrivateColumnMap] = PrivateColumnMap,
    ) -> None:
        self.path = path
        self.column_map = column_map

    def read(self) -> pd.DataFrame:
        engine = ExcelEngine.for_path(self.path)
        workbook = pd.ExcelFile(self.path, engine=engine)
        required = self.column_map.source_columns()
        last_error: Exception | None = None
        for sheet_name in workbook.sheet_names:
            raw = pd.read_excel(
                self.path, sheet_name=sheet_name, header=None, engine=engine
            )
            try:
                header_row = self._find_header_row(raw, required)
            except ValueError as exc:
                last_error = exc
                continue
            frame = raw.iloc[header_row + 1 :].copy()
            frame.columns = [str(col).strip() for col in raw.iloc[header_row].tolist()]
            return frame.reset_index(drop=True)
        raise ValueError(str(last_error) if last_error else "No usable Laguna sheet found")

    @staticmethod
    def _find_header_row(raw: pd.DataFrame, required: tuple[str, ...]) -> int:
        needed = {_normalize_name(name) for name in required}
        for idx, row in raw.iterrows():
            values = {_normalize_name(val) for val in row.tolist() if pd.notna(val)}
            if needed.issubset(values):
                return int(idx)
        raise ValueError(
            "Could not find a Laguna header row containing: " + ", ".join(required)
        )


class UnitConverter:
    """Convert Laguna source units to the ASHRAE °F / psi scale.

    Runs on the raw table before mapping. Derived ``P_lift`` then subtracts
    psi from psi. ``CDW_DT`` and ``CHW_DT`` subtract already-converted
    absolute temperatures, so the +32 cancels and those differences stay in °F.
    """

    # °F = °C * 1.8 + 32
    ABSOLUTE_TEMP_C: tuple[str, ...] = (
        "ECW",
        "LCW",
        "ECDW",
        "LCDW",
        "ERT",
        "CRT",
        "CMPD",
        "OILT",
    )
    # °F = °C * 1.8. Differences must not add 32.
    DELTA_TEMP_C: tuple[str, ...] = ("SUPRHEAT", "COND_APP")
    # psi = kPa / 6.89476
    PRESSURE_KPA: tuple[str, ...] = ("ERP", "CRP", "OILPD")

    def convert(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        lookup = self._lookup(out.columns)
        for name in self.ABSOLUTE_TEMP_C:
            col = self._require(lookup, name)
            out[col] = pd.to_numeric(out[col], errors="coerce") * C_TO_F_SCALE + C_TO_F_OFFSET
        for name in self.DELTA_TEMP_C:
            col = self._require(lookup, name)
            out[col] = pd.to_numeric(out[col], errors="coerce") * C_TO_F_SCALE
        for name in self.PRESSURE_KPA:
            col = self._require(lookup, name)
            out[col] = pd.to_numeric(out[col], errors="coerce") / KPA_PER_PSI
        return out

    @staticmethod
    def _lookup(columns: pd.Index) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for col in columns:
            lookup.setdefault(_normalize_name(col), str(col))
        return lookup

    @staticmethod
    def _require(lookup: dict[str, str], name: str) -> str:
        key = _normalize_name(name)
        if key not in lookup:
            raise KeyError(f"Laguna file is missing unit-conversion column {name!r}")
        return lookup[key]


class SchemaMapper:
    """Project a Laguna table onto the ASHRAE schema column order."""

    def __init__(
        self,
        column_map: type[PrivateColumnMap] = PrivateColumnMap,
        *,
        time_step_seconds: float = TIME_STEP_SECONDS,
        trailing_rows_to_drop: int = TRAILING_ROWS_TO_DROP,
        source_file_value: str = SOURCE_FILE_VALUE,
    ) -> None:
        self.column_map = column_map
        self.time_step_seconds = time_step_seconds
        self.trailing_rows_to_drop = trailing_rows_to_drop
        self.source_file_value = source_file_value

    def map(self, source: pd.DataFrame, schema_columns: list[str]) -> pd.DataFrame:
        lookup = self._column_lookup(source.columns)
        out = pd.DataFrame(index=range(len(source)), columns=schema_columns)
        mapped = self._map_columns(source, lookup)

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
        out = self._drop_trailing(out)

        time_col = SchemaReader.find(schema_columns, "time")
        out[time_col] = np.arange(len(out), dtype="float64") * self.time_step_seconds

        source_col = SchemaReader.find(schema_columns, "source_file")
        out[source_col] = self.source_file_value
        return out.loc[:, schema_columns]

    def _map_columns(
        self, source: pd.DataFrame, lookup: dict[str, str]
    ) -> dict[str, pd.Series]:
        mapped: dict[str, pd.Series] = {}
        for dest, src_name in self.column_map.DIRECT.items():
            mapped[dest] = self._numeric_series(source, lookup, src_name)

        for dest, (left, right, op) in self.column_map.DERIVED.items():
            mapped[dest] = self.column_map.apply_operator(
                self._numeric_series(source, lookup, left),
                self._numeric_series(source, lookup, right),
                op,
            )

        # Absolute temps are already °F, so these differences are °F.
        mapped["CDW_DT"] = mapped["TCO"] - mapped["TCI"]
        mapped["CHW_DT"] = mapped["TEI"] - mapped["TEO"]
        return mapped

    def _drop_trailing(self, frame: pd.DataFrame) -> pd.DataFrame:
        n_drop = self.trailing_rows_to_drop
        if len(frame) <= n_drop:
            raise ValueError(
                f"Only {len(frame)} rows after mapping; cannot drop {n_drop} trailing rows"
            )
        return frame.iloc[:-n_drop].reset_index(drop=True)

    @staticmethod
    def _column_lookup(columns: pd.Index) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for col in columns:
            lookup.setdefault(_normalize_name(col), str(col))
        return lookup

    @staticmethod
    def _require_source_column(lookup: dict[str, str], name: str) -> str:
        key = _normalize_name(name)
        if key not in lookup:
            raise KeyError(f"Laguna file is missing source column {name!r}")
        return lookup[key]

    def _numeric_series(
        self, frame: pd.DataFrame, lookup: dict[str, str], name: str
    ) -> pd.Series:
        col = self._require_source_column(lookup, name)
        return pd.to_numeric(frame[col], errors="coerce")


class PrivatePreprocessor:
    """Read a Laguna export, align it to the ASHRAE schema, write parquet."""

    def __init__(
        self,
        source_path: Path = DEFAULT_SOURCE,
        schema_path: Path = DEFAULT_SCHEMA,
        out_path: Path = DEFAULT_OUT,
        *,
        column_map: type[PrivateColumnMap] = PrivateColumnMap,
    ) -> None:
        self.source_path = source_path
        self.schema_path = schema_path
        self.out_path = out_path
        self.column_map = column_map

    def run(self) -> pd.DataFrame:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
        if not self.schema_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        schema_columns = SchemaReader(self.schema_path).columns()
        source = LagunaReader(self.source_path, self.column_map).read()
        source = UnitConverter().convert(source)
        frame = SchemaMapper(self.column_map).map(source, schema_columns)

        os.makedirs(self.out_path.parent, exist_ok=True)
        frame.to_parquet(self.out_path, engine="pyarrow", compression="snappy", index=False)
        return frame
