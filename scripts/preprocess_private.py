"""CLI: map a private Laguna export onto the ASHRAE schema and write parquet.

Steps in this script
--------------------
1. Parse ``--source``, ``--schema``, and ``--out``.
2. Run ``PrivatePreprocessor``: read, convert units, map, write.
   - Absolute °C→°F (×1.8+32): ECW, LCW, ECDW, LCDW, ERT, CRT, CMPD, OILT.
   - Difference °C→°F (×1.8 only): SUPRHEAT, COND_APP.
   - Pressure kPa→psi (÷6.89476): ERP, CRP, OILPD.
3. Print shape, schema order, populated maps, unmapped features, and time range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chiller_fault_detection.data.load import FEATURE_COLUMNS
from chiller_fault_detection.data.preprocess_private import (
    DEFAULT_OUT,
    DEFAULT_SCHEMA,
    DEFAULT_SOURCE,
    PrivateColumnMap,
    PrivatePreprocessor,
    SchemaReader,
    UnitConverter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    frame = PrivatePreprocessor(args.source, args.schema, args.out).run()
    schema_columns = SchemaReader(args.schema).columns()
    mapped_columns = PrivateColumnMap.mapped_feature_columns()
    missing_features = [col for col in FEATURE_COLUMNS if col not in mapped_columns]
    populated = [col for col in mapped_columns if frame[col].notna().any()]

    print(
        "converted "
        f"C_to_F+32={list(UnitConverter.ABSOLUTE_TEMP_C)} "
        f"C_to_F={list(UnitConverter.DELTA_TEMP_C)} "
        f"kPa_to_psi={list(UnitConverter.PRESSURE_KPA)}"
    )
    print(f"Wrote {len(frame)} rows × {len(frame.columns)} cols → {args.out}")
    print(
        f"schema_columns={len(schema_columns)} "
        f"order_match={list(frame.columns) == schema_columns}"
    )
    print(f"mapped_populated={len(populated)}/{len(mapped_columns)}: {populated}")
    print(f"feature_columns_nan={missing_features}")
    print(f"source_file={frame['source_file'].unique().tolist()}")
    print(f"time_range=[{frame['time'].iloc[0]}, {frame['time'].iloc[-1]}]")


if __name__ == "__main__":
    main()
