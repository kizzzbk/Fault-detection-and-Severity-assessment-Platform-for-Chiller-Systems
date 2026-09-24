"""CLI: load ASHRAE RP-1043 Excel tests into dataset.parquet."""

import argparse
import json
from pathlib import Path

from chiller_fault_detection.data.load import load_dataset, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df, report = load_dataset(args.data_dir)
    out = save_parquet(df, report, path=args.out)
    print(
        f"Wrote {report.loaded_rows} rows from {report.loaded_files} files → {out}"
    )
    print(
        json.dumps(
            {
                "dropped_nan_rows": report.dropped_nan_rows,
                "replaced_tiny_values": report.replaced_tiny_values,
                "skipped_files": len(report.skipped_files),
                "fault_counts": report.fault_counts,
                "path_label_counts": report.path_label_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
