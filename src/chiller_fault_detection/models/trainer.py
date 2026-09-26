"""Train TS-TCC on windowed RP-1043 splits and evaluate fault_id."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from chiller_fault_detection.config import RANDOM_SEED, SCALER_PATH, SPLITS_DIR
from chiller_fault_detection.data.load import FEATURE_COLUMNS
from chiller_fault_detection.data.preprocess import (
    load_dataset,
    partition_by_split,
    sort_within_files,
    time_splits,
)
from chiller_fault_detection.labels import ID_TO_LABEL
from chiller_fault_detection.models.TC import TC
from chiller_fault_detection.models.augmentations import DataTransform
from chiller_fault_detection.models.loss import NTXentLoss, SupConLoss
from chiller_fault_detection.models.model import base_Model

LOGGER = logging.getLogger("ts_tcc")


@dataclass
class AugmentConfig:
    # Standardized features: keep jitter small (paper, Section 3.1).
    weak_jitter_ratio: float = 0.05
    jitter_scale_ratio: float = 0.1
    jitter_ratio: float = 0.08
    max_seg: int = 8


@dataclass
class ContextConfig:
    temperature: float = 0.2
    use_cosine_similarity: bool = True


@dataclass
class TemporalConfig:
    hidden_dim: int = 100
    timesteps: int = 6


@dataclass
class TrainConfig:
    input_channels: int = len(FEATURE_COLUMNS)
    kernel_size: int = 8
    stride: int = 1
    final_out_channels: int = 128
    num_classes: int = len(ID_TO_LABEL)
    dropout: float = 0.35
    features_len: int = 1
    num_epoch: int = 20
    linear_epoch: int = 20
    beta1: float = 0.9
    beta2: float = 0.99
    lr: float = 3e-4
    drop_last: bool = True
    batch_size: int = 128
    window: int = 128
    stride_windows: int = 64
    seed: int = RANDOM_SEED
    augmentation: AugmentConfig = None
    Context_Cont: ContextConfig = None
    TC: TemporalConfig = None

    def __post_init__(self) -> None:
        if self.augmentation is None:
            self.augmentation = AugmentConfig()
        if self.Context_Cont is None:
            self.Context_Cont = ContextConfig()
        if self.TC is None:
            self.TC = TemporalConfig()
        self.features_len = encoder_length(self.window, self.kernel_size, self.stride)
        if self.features_len <= self.TC.timesteps:
            raise ValueError(
                f"Encoder length {self.features_len} must exceed TC timesteps {self.TC.timesteps}"
            )


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, aug1=None, aug2=None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.aug1 = aug1
        self.aug2 = aug2

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]
        if self.aug1 is None:
            return x, y, x, x
        return x, y, self.aug1[index], self.aug2[index]


def encoder_length(length: int, kernel_size: int, stride: int) -> int:
    """Temporal length after the three conv blocks in ``base_Model``."""

    def _conv(size: int, kernel: int, step: int, padding: int) -> int:
        return (size + 2 * padding - kernel) // step + 1

    def _pool(size: int) -> int:
        return (size + 2 - 2) // 2 + 1

    length = _pool(_conv(length, kernel_size, stride, kernel_size // 2))
    length = _pool(_conv(length, 8, 1, 4))
    length = _pool(_conv(length, 8, 1, 4))
    return int(length)


def window_part(frame, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for _, group in frame.groupby("source_file", sort=False):
        values = group.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        label = int(group["fault_id"].iloc[0])
        if len(values) < window:
            continue
        for start in range(0, len(values) - window + 1, stride):
            xs.append(values[start : start + window].T)
            ys.append(label)
    if not xs:
        raise ValueError(f"No windows of length {window} could be built.")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def load_window_splits(config: TrainConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    df = sort_within_files(load_dataset())
    df = time_splits(df)
    parts = partition_by_split(df)
    scaler = joblib.load(SCALER_PATH)
    splits = {}
    for name, part in parts.items():
        scaled = part.copy()
        scaled.loc[:, FEATURE_COLUMNS] = scaler.transform(part.loc[:, FEATURE_COLUMNS])
        splits[name] = window_part(scaled, config.window, config.stride_windows)
    return splits


def make_loaders(
    splits,
    config: TrainConfig,
    training_mode: str,
    *,
    train_xy: tuple[np.ndarray, np.ndarray] | None = None,
    batch_size: int | None = None,
):
    x_train, y_train = train_xy if train_xy is not None else splits["train"]
    aug1 = aug2 = None
    if training_mode in ("self_supervised", "SupCon"):
        LOGGER.info(
            "Augmenting %d train windows before %s",
            len(y_train),
            training_mode,
        )
        aug1, aug2 = DataTransform(x_train, config)

    batch = batch_size or config.batch_size

    def _loader(name: str, shuffle: bool, drop_last: bool, xy=None) -> DataLoader:
        x, y = xy if xy is not None else splits[name]
        dataset = WindowDataset(
            x,
            y,
            aug1 if name == "train" else None,
            aug2 if name == "train" else None,
        )
        use_batch = min(batch, len(dataset))
        return DataLoader(
            dataset,
            batch_size=use_batch,
            shuffle=shuffle,
            drop_last=drop_last and len(dataset) > use_batch,
            num_workers=0,
        )

    return (
        _loader("train", True, config.drop_last, (x_train, y_train)),
        _loader("val", False, False),
        _loader("test", False, False),
    )


def stratified_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n_take = min(len(idx), max(1, int(round(len(idx) * fraction))))
        chosen.append(rng.choice(idx, size=n_take, replace=False))
    return np.sort(np.concatenate(chosen))


def _unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def pseudo_labels(model, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits, _ = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def Trainer(
    model,
    temporal_contr_model,
    model_optimizer,
    temp_cont_optimizer,
    train_dl,
    valid_dl,
    device,
    config: TrainConfig,
    training_mode: str,
    epochs: int,
):
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = model_train(
            model,
            temporal_contr_model,
            model_optimizer,
            temp_cont_optimizer,
            criterion,
            train_dl,
            config,
            device,
            training_mode,
        )
        valid_loss, valid_acc, _, _ = model_evaluate(
            model, temporal_contr_model, valid_dl, device, training_mode
        )
        if training_mode not in ("self_supervised", "SupCon"):
            scheduler.step(valid_loss)
        LOGGER.info(
            "epoch %d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
        )


def model_train(
    model,
    temporal_contr_model,
    model_optimizer,
    temp_cont_optimizer,
    criterion,
    train_loader,
    config,
    device,
    training_mode,
):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for data, labels, aug1, aug2 in train_loader:
        data = data.float().to(device)
        labels = labels.long().to(device)
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)

        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            _, features1 = model(aug1)
            _, features2 = model(aug2)
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            temp_cont_loss1, zis = temporal_contr_model(features1, features2)
            temp_cont_loss2, zjs = temporal_contr_model(features2, features1)
            nt_xent = NTXentLoss(
                device,
                config.batch_size,
                config.Context_Cont.temperature,
                config.Context_Cont.use_cosine_similarity,
            )
            loss = (temp_cont_loss1 + temp_cont_loss2) + 0.7 * nt_xent(zis, zjs)
        elif training_mode == "SupCon":
            _, features1 = model(aug1)
            _, features2 = model(aug2)
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            temp_cont_loss1, feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, feat2 = temporal_contr_model(features2, features1)
            sup_features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            loss = 0.01 * (temp_cont_loss1 + temp_cont_loss2) + 0.1 * SupConLoss(device)(
                sup_features, labels
            )
        else:
            predictions, _ = model(data)
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        if training_mode in ("self_supervised", "SupCon"):
            temp_cont_optimizer.step()

    mean_loss = float(torch.tensor(total_loss).mean())
    if training_mode in ("self_supervised", "SupCon"):
        return mean_loss, 0.0
    return mean_loss, float(torch.tensor(total_acc).mean())


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_acc = []
    outs = []
    trgs = []

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data = data.float().to(device)
            labels = labels.long().to(device)
            if training_mode in ("self_supervised", "SupCon"):
                continue
            predictions, _ = model(data)
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())
            outs.append(predictions.argmax(dim=1).cpu().numpy())
            trgs.append(labels.cpu().numpy())

    if training_mode in ("self_supervised", "SupCon"):
        return 0.0, 0.0, np.array([]), np.array([])
    return (
        float(torch.tensor(total_loss).mean()),
        float(torch.tensor(total_acc).mean()),
        np.concatenate(outs),
        np.concatenate(trgs),
    )


def _optimizers(model, temporal_contr_model, config: TrainConfig):
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=3e-4,
    )
    temp_cont_optimizer = torch.optim.Adam(
        temporal_contr_model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=3e-4,
    )
    return model_optimizer, temp_cont_optimizer


def _freeze_encoder(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "logits" in name


def _report(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    names = [ID_TO_LABEL[i].value for i in range(len(ID_TO_LABEL))]
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    text = classification_report(
        y_true, y_pred, target_names=names, digits=4, zero_division=0
    )
    summary = f"{split_name}: accuracy={acc:.4f}  macro_f1={macro:.4f}\n{text}"
    LOGGER.info("\n%s", summary)
    return summary


def run(config: TrainConfig, out_dir: Path, device: torch.device) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_window_splits(config)
    for name, (x, y) in splits.items():
        LOGGER.info("%s windows: %s  labels: %s", name, x.shape, np.bincount(y))

    model = base_Model(config).to(device)
    temporal = TC(config, device).to(device)

    train_dl, valid_dl, test_dl = make_loaders(splits, config, "self_supervised")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("Self-supervised pretraining")
    Trainer(
        model,
        temporal,
        model_opt,
        temporal_opt,
        train_dl,
        valid_dl,
        device,
        config,
        "self_supervised",
        config.num_epoch,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict": temporal.state_dict(),
        "config": {
            "window": config.window,
            "stride_windows": config.stride_windows,
            "features_len": config.features_len,
            "num_classes": config.num_classes,
        },
    }
    ckpt_path = out_dir / "ckp_last.pt"
    torch.save(checkpoint, ckpt_path)
    LOGGER.info("Wrote encoder checkpoint → %s", ckpt_path)

    _freeze_encoder(model)
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "train_linear")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("Linear classifier on frozen encoder")
    Trainer(
        model,
        temporal,
        model_opt,
        temporal_opt,
        train_dl,
        valid_dl,
        device,
        config,
        "train_linear",
        config.linear_epoch,
    )

    _, _, val_pred, val_true = model_evaluate(
        model, temporal, valid_dl, device, "train_linear"
    )
    _, _, test_pred, test_true = model_evaluate(
        model, temporal, test_dl, device, "train_linear"
    )
    report = _report("val", val_true, val_pred) + "\n" + _report("test", test_true, test_pred)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "linear_head.pt")
    LOGGER.info("Wrote report → %s", report_path)


def run_ca_tcc(config: TrainConfig, out_dir: Path, device: torch.device, label_fraction: float = 0.01) -> None:
    """CA-TCC: TS-TCC pretrain, fine-tune on 1% labels, pseudo-labels, SupCon, linear eval."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_window_splits(config)
    x_train, y_train = splits["train"]
    few_idx = stratified_indices(y_train, label_fraction, config.seed)
    few_xy = (x_train[few_idx], y_train[few_idx])
    LOGGER.info("Labeled fraction %.0f%%: %d / %d windows", label_fraction * 100, len(few_idx), len(y_train))

    model = base_Model(config).to(device)
    temporal = TC(config, device).to(device)

    train_dl, valid_dl, test_dl = make_loaders(splits, config, "self_supervised")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 1: self-supervised pretraining")
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "self_supervised", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ssl.pt")

    _unfreeze(model)
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "ft_1p", train_xy=few_xy, batch_size=16)
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 2: fine-tune on %.0f%% labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "ft_1p", config.linear_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ft.pt")

    y_pseudo = pseudo_labels(model, x_train, device, config.batch_size)
    agree = float((y_pseudo == y_train).mean())
    LOGGER.info("CA-TCC stage 3: pseudo labels agree with true labels on %.3f of train", agree)
    np.save(out_dir / "pseudo_labels.npy", y_pseudo)

    _unfreeze(model)
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "SupCon", train_xy=(x_train, y_pseudo)
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 4: SupCon on pseudo labels")
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "SupCon", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_supcon.pt")

    _freeze_encoder(model)
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "train_linear", train_xy=few_xy, batch_size=16
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 5: linear head on %.0f%% true labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "train_linear", config.linear_epoch)

    _, _, val_pred, val_true = model_evaluate(model, temporal, valid_dl, device, "train_linear")
    _, _, test_pred, test_true = model_evaluate(model, temporal, test_dl, device, "train_linear")
    report = (
        f"label_fraction={label_fraction} n_labeled={len(few_idx)} pseudo_agree={agree:.4f}\n"
        + _report("val", val_true, val_pred)
        + "\n"
        + _report("test", test_true, test_pred)
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "linear_head.pt")
    LOGGER.info("Wrote report → %s", report_path)


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
    out_dir = args.out_dir or (SPLITS_DIR.parent / ("ca_tcc" if args.method == "ca-tcc" else "ts_tcc"))
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
