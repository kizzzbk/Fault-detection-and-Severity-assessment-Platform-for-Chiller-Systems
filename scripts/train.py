"""CLI: train TS-TCC or CA-TCC with the CNN or DINOv3 encoder."""

import argparse
import logging
from pathlib import Path

import torch

from chiller_fault_detection.config import ARTIFACTS_DIR
from chiller_fault_detection.models.trainer import (
    Dinov3Config,
    TrainConfig,
    run,
    run_ca_tcc,
)

METHODS = ("ts-tcc", "ca-tcc")


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="default 16 for dinov3, 128 for cnn",
    )
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--method", choices=METHODS, default="ca-tcc")
    parser.add_argument(
        "--encoder",
        choices=("cnn", "dinov3"),
        default="cnn",
        help="cnn = TS-TCC conv; dinov3 = pretrained DINOv3 ViT",
    )
    parser.add_argument("--label-fraction", type=float, default=0.01)
    parser.add_argument(
        "--finetune-encoder",
        action="store_true",
        help="train the full encoder. For dinov3 this also unfreezes the ViT from stage 1 at lr 1e-5",
    )
    parser.add_argument(
        "--dinov3-variant",
        choices=("vits16", "vits16plus", "vitb16", "vitl16"),
        default="vits16",
        help="pretrained DINOv3 ViT. Ignored unless --encoder dinov3",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="square image the window is resized to. Must be a multiple of 16. Ignored unless --encoder dinov3",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device_name = args.device or _default_device()
    logging.info("Device: %s", device_name)
    run_name = _run_name(args.method, args.encoder)
    if args.finetune_encoder:
        run_name += "_ft"
    out_dir = args.out_dir or (ARTIFACTS_DIR / run_name)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16 if args.encoder == "dinov3" else 128

    config = TrainConfig(
        encoder=args.encoder,
        num_epoch=args.epochs,
        linear_epoch=args.linear_epochs,
        batch_size=batch_size,
        window=args.window,
        stride_windows=args.stride,
        dinov3=Dinov3Config(variant=args.dinov3_variant, img_size=args.img_size),
        finetune_encoder=args.finetune_encoder,
    )
    device = torch.device(device_name)
    if args.method == "ca-tcc":
        run_ca_tcc(config, out_dir, device, args.label_fraction)
    else:
        run(config, out_dir, device)


if __name__ == "__main__":
    main()
