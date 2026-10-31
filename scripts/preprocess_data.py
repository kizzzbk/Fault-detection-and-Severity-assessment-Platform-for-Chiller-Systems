"""CLI: time-split + scale processed dataset.parquet."""
import argparse
import json
from pathlib import Path

from chiller_fault_detection.config import (
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from chiller_fault_detection.data.preprocess import run_preprocess


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
