"""CLI: train TS-TCC or CA-TCC and write the report under artifacts."""

import argparse
import logging
from pathlib import Path

import torch

from chiller_fault_detection.config import SPLITS_DIR
from chiller_fault_detection.models.trainer import TrainConfig, run, run_ca_tcc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--method", choices=("ts-tcc", "ca-tcc"), default="ts-tcc")
    parser.add_argument("--label-fraction", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or (
        SPLITS_DIR.parent / ("ca_tcc" if args.method == "ca-tcc" else "ts_tcc")
    )
    config = TrainConfig(
        num_epoch=args.epochs,
        linear_epoch=args.linear_epochs,
        batch_size=args.batch_size,
        window=args.window,
        stride_windows=args.stride,
    )
    if args.method == "ca-tcc":
        run_ca_tcc(config, out_dir, torch.device(device_name), args.label_fraction)
    else:
        run(config, out_dir, torch.device(device_name))


if __name__ == "__main__":
    main()