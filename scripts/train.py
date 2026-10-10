"""CLI: train TS-TCC, CA-TCC or CA-TCC-FixMatch and write the report under artifacts."""

import argparse
import logging
from pathlib import Path

import torch

from chiller_fault_detection.config import ARTIFACTS_DIR
from chiller_fault_detection.models.fixmatch import FixMatchConfig, run_ca_tcc_fixmatch
from chiller_fault_detection.models.trainer import (
    PatchTSTConfig,
    TrainConfig,
    run,
    run_ca_tcc,
)

METHODS = ("ts-tcc", "ca-tcc", "ca-tcc-fixmatch")


def _run_name(method: str, encoder: str) -> str:
    return f"{method.replace('-', '_')}_{encoder}"


def _default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        return "cpu"
    return "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--method", choices=METHODS, default="ca-tcc")
    parser.add_argument(
        "--encoder",
        choices=("cnn", "patchtst"),
        default="cnn",
        help="cnn = original TS-TCC conv encoder; patchtst = PatchTST replacement",
    )
    parser.add_argument("--label-fraction", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument(
        "--fixmatch-tau",
        type=float,
        default=0.95,
        help="ca-tcc-fixmatch only: keep pseudo labels with max softmax above this",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device_name = args.device or _default_device()
    logging.info("Device: %s", device_name)
    out_dir = args.out_dir or (ARTIFACTS_DIR / _run_name(args.method, args.encoder))

    patchtst = PatchTSTConfig()
    if args.patch_len is not None:
        patchtst.patch_len = args.patch_len
    if args.patch_stride is not None:
        patchtst.patch_stride = args.patch_stride
    if args.d_model is not None:
        patchtst.d_model = args.d_model
    if args.n_layers is not None:
        patchtst.n_layers = args.n_layers
    if args.n_heads is not None:
        patchtst.n_heads = args.n_heads
    if args.d_ff is not None:
        patchtst.d_ff = args.d_ff

    config = TrainConfig(
        encoder=args.encoder,
        num_epoch=args.epochs,
        linear_epoch=args.linear_epochs,
        batch_size=args.batch_size,
        window=args.window,
        stride_windows=args.stride,
        patchtst=patchtst,
    )
    device = torch.device(device_name)
    if args.method == "ca-tcc-fixmatch":
        fixmatch = FixMatchConfig(tau=args.fixmatch_tau)
        run_ca_tcc_fixmatch(config, fixmatch, out_dir, device, args.label_fraction)
    elif args.method == "ca-tcc":
        run_ca_tcc(config, out_dir, device, args.label_fraction)
    else:
        run(config, out_dir, device)


if __name__ == "__main__":
    main()
