"""Train TS-TCC on windowed RP-1043 splits and evaluate fault_id."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chiller_fault_detection.config import RANDOM_SEED, SPLITS_DIR
from chiller_fault_detection.data.load import MODEL_FEATURE_COLUMNS
from chiller_fault_detection.labels import ID_TO_LABEL
from chiller_fault_detection.models.augmentations import DataTransform
from chiller_fault_detection.models.dinov3 import Dinov3Encoder, n_dino_tokens
from chiller_fault_detection.models.loss import (
    MaskedReconstructionLoss,
    NTXentLoss,
    SupConLoss,
)
from chiller_fault_detection.models.model import base_Model
from chiller_fault_detection.models.TC import TC

LOGGER = logging.getLogger("ts_tcc")


@dataclass
class AugmentConfig:
    # Weak view jitters then scales; CNN strong view permutes then jitters.
    # Scale/jitter match the HAR config in the TS-TCC repo (sigma 1.1 and 0.8).
    # DINOv3 strong view uses milder jitter + masking, not permute.
    weak_jitter_ratio: float = 0.05
    jitter_scale_ratio: float = 1.1
    jitter_ratio: float = 0.8
    max_seg: int = 8
    dinov3_jitter_ratio: float = 0.2
    dinov3_mask_ratio: float = 0.4


@dataclass
class ContextConfig:
    temperature: float = 0.2
    use_cosine_similarity: bool = True


@dataclass
class TemporalConfig:
    hidden_dim: int = 100
    timesteps: int = 6


@dataclass
class Dinov3Config:
    variant: str = "vits16"
    img_size: int = 224
    patch_size: int = 16
    mask_patch_len: int = 8
    mask_patch_stride: int = 8


@dataclass
class TrainConfig:
    encoder: str = "cnn"
    input_channels: int = len(MODEL_FEATURE_COLUMNS)
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
    finetune_encoder: bool = False
    augmentation: AugmentConfig = None
    Context_Cont: ContextConfig = None
    TC: TemporalConfig = None
    dinov3: Dinov3Config = None

    def __post_init__(self) -> None:
        if self.augmentation is None:
            self.augmentation = AugmentConfig()
        if self.Context_Cont is None:
            self.Context_Cont = ContextConfig()
        if self.TC is None:
            self.TC = TemporalConfig()
        if self.dinov3 is None:
            self.dinov3 = Dinov3Config()
        if self.encoder == "cnn":
            self.features_len = encoder_length(self.window, self.kernel_size, self.stride)
        elif self.encoder == "dinov3":
            self.features_len = n_dino_tokens(self.dinov3.img_size, self.dinov3.patch_size)
        else:
            raise ValueError(f"Unknown encoder {self.encoder!r}; use 'cnn' or 'dinov3'")
        if self.features_len <= self.TC.timesteps:
            raise ValueError(
                f"Encoder length {self.features_len} must exceed TC timesteps {self.TC.timesteps}"
            )


def build_encoder(config: TrainConfig) -> nn.Module:
    """CNN or DINOv3. Each returns ``(logits, features)``."""
    if config.encoder == "cnn":
        return base_Model(config)
    if config.encoder == "dinov3":
        model = Dinov3Encoder(config)
        if config.finetune_encoder:
            LOGGER.info("DINOv3 backbone stays trainable at lr %.1e", _encoder_lr(config))
        else:
            model.freeze_backbone()
            LOGGER.info(
                "DINOv3 backbone frozen from stage 1; projection stays trainable until the linear probe"
            )
        return model
    raise ValueError(f"Unknown encoder {config.encoder!r}")


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


def window_part(
    frame, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    sevs: list[int] = []
    for _, group in frame.groupby("source_file", sort=False):
        values = group.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        label = int(group["fault_id"].iloc[0])
        severity = int(group["severity_level"].iloc[0])
        if len(values) < window:
            continue
        for start in range(0, len(values) - window + 1, stride):
            xs.append(values[start : start + window].T)
            ys.append(label)
            sevs.append(severity)
    if not xs:
        raise ValueError(f"No windows of length {window} could be built.")
    return (
        np.stack(xs),
        np.asarray(ys, dtype=np.int64),
        np.asarray(sevs, dtype=np.int64),
    )


def load_window_splits(
    config: TrainConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load scaled splits and cut windows. Severity is copied from each source file."""
    needed = ["source_file", "fault_id", "severity_level", *MODEL_FEATURE_COLUMNS]
    splits = {}
    for name in ("train", "val", "test"):
        parquet_path = SPLITS_DIR / f"X_{name}.parquet"
        csv_path = SPLITS_DIR / f"X_{name}.csv"
        if parquet_path.is_file():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.is_file():
            frame = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Split frame not found: {parquet_path} (or {csv_path})")
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"X_{name} is missing columns needed to window: {missing}")
        splits[name] = window_part(frame, config.window, config.stride_windows)
    return splits


def make_loaders(
    splits,
    config: TrainConfig,
    training_mode: str,
    *,
    train_xy: tuple[np.ndarray, np.ndarray] | None = None,
    batch_size: int | None = None,
):
    x_train, y_train = (train_xy if train_xy is not None else splits["train"])[:2]
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
        x, y = (xy if xy is not None else splits[name])[:2]
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
        n_take = min(len(idx), max(1, round(len(idx) * fraction)))
        chosen.append(rng.choice(idx, size=n_take, replace=False))
    return np.sort(np.concatenate(chosen))


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
        valid_loss, valid_acc, *_ = model_evaluate(
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
            if config.encoder == "dinov3":
                _, features1 = model(aug1)
                # Apply 16x16 patch masking directly on the 2D image plane for aug2
                _, features2, patch_mask = model(
                    aug2, mask_ratio=config.augmentation.dinov3_mask_ratio
                )
                recon_loss = MaskedReconstructionLoss()(features2, features1, patch_mask)
            else:
                _, features1 = model(aug1)
                _, features2 = model(aug2)
                recon_loss = 0.0

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            temp_cont_loss1, zis = temporal_contr_model(features1, features2)
            temp_cont_loss2, zjs = temporal_contr_model(features2, features1)
            temp_temperature = (
                0.07 if config.encoder == "dinov3" else config.Context_Cont.temperature
            )
            nt_xent = NTXentLoss(
                device,
                config.batch_size,
                temp_temperature,
                config.Context_Cont.use_cosine_similarity,
            )
            if config.encoder == "dinov3":
                loss = (
                    0.1 * (temp_cont_loss1 + temp_cont_loss2)
                    + 0.7 * nt_xent(zis, zjs)
                    + 1.0 * recon_loss
                )
            else:
                loss = (temp_cont_loss1 + temp_cont_loss2) + 0.7 * nt_xent(zis, zjs)
        elif training_mode == "SupCon":
            _, features1 = model(aug1)
            _, features2 = model(aug2)
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            temp_cont_loss1, feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, feat2 = temporal_contr_model(features2, features1)
            sup_features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            sup_temp = 0.07 if config.encoder == "dinov3" else 0.2
            if config.encoder == "dinov3":
                loss = SupConLoss(device, temperature=sup_temp)(sup_features, labels)
            else:
                loss = 0.01 * (temp_cont_loss1 + temp_cont_loss2) + 0.1 * SupConLoss(
                    device, temperature=sup_temp
                )(sup_features, labels)
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
    probs = []

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data = data.float().to(device)
            labels = labels.long().to(device)
            if training_mode in ("self_supervised", "SupCon"):
                continue
            logits, _ = model(data)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            total_acc.append(labels.eq(pred).float().mean())
            total_loss.append(loss.item())
            outs.append(pred.cpu().numpy())
            trgs.append(labels.cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())

    if training_mode in ("self_supervised", "SupCon"):
        empty = np.array([])
        return 0.0, 0.0, empty, empty, empty
    return (
        float(torch.tensor(total_loss).mean()),
        float(torch.tensor(total_acc).mean()),
        np.concatenate(outs),
        np.concatenate(trgs),
        np.concatenate(probs),
    )


def _encoder_lr(config: TrainConfig) -> float:
    """A trainable DINOv3 ViT caps at 1e-5.

    A frozen DINOv3 backbone leaves the projection and the head on ``config.lr``
    (3e-4). The same rate is also used for the temporal-contrasting module.
    """
    if config.encoder == "dinov3" and config.finetune_encoder:
        return min(config.lr, 1e-5)
    return config.lr


def _optimizers(model, temporal_contr_model, config: TrainConfig):
    lr = _encoder_lr(config)
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(config.beta1, config.beta2),
        weight_decay=3e-4,
    )
    temp_cont_optimizer = torch.optim.Adam(
        temporal_contr_model.parameters(),
        lr=lr,
        betas=(config.beta1, config.beta2),
        weight_decay=3e-4,
    )
    return model_optimizer, temp_cont_optimizer


def _freeze_encoder(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "logits" in name


def _maybe_freeze_for_linear(model, config: TrainConfig) -> None:
    """Linear probe for CNN and DINOv3. ``finetune_encoder`` trains the full net.

    DINOv3's ViT is already frozen in ``build_encoder``. This second freeze
    also stops the feature projection, so the last stage updates only ``logits``.
    """
    if config.finetune_encoder:
        LOGGER.info("Supervised fine-tune of full %s encoder + head", config.encoder)
        return
    _freeze_encoder(model)
    LOGGER.info("Linear classifier on frozen encoder")


def _class_names() -> list[str]:
    return [ID_TO_LABEL[i].value for i in range(len(ID_TO_LABEL))]


def _experiment_header(config: TrainConfig, method: str, extra: str = "") -> str:
    lines = [
        f"method={method}",
        f"encoder={config.encoder}",
        f"window={config.window}",
        f"features_len={config.features_len}",
        f"final_out_channels={config.final_out_channels}",
        f"finetune_encoder={config.finetune_encoder}",
    ]
    if config.encoder == "dinov3":
        d3 = config.dinov3
        lines.append(
            "dinov3 "
            f"variant={d3.variant} img_size={d3.img_size} patch_size={d3.patch_size} "
            f"n_tokens={config.features_len} lr={_encoder_lr(config):.1e} "
            f"freeze_backbone={not config.finetune_encoder}"
        )
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n\n"


def _report(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    names = _class_names()
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    text = classification_report(
        y_true, y_pred, target_names=names, digits=4, zero_division=0
    )
    summary = f"{split_name}: accuracy={acc:.4f}  macro_f1={macro:.4f}\n{text}"
    LOGGER.info("\n%s", summary)
    return summary


def _heatmap(matrix, names: list[str], path: Path, title: str, fmt: str, vmin, vmax) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_eval(
    out_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    names = _class_names()
    np.save(out_dir / f"{split_name}_y_true.npy", y_true)
    np.save(out_dir / f"{split_name}_y_pred.npy", y_pred)
    np.save(out_dir / f"{split_name}_y_prob.npy", y_prob)

    labels = list(range(len(names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        out_dir / f"{split_name}_confusion_matrix.csv"
    )
    _heatmap(
        cm,
        names,
        out_dir / f"{split_name}_confusion_matrix.png",
        f"{split_name} confusion matrix",
        "d",
        None,
        None,
    )

    mean_prob = np.zeros((len(names), len(names)), dtype=np.float64)
    for k in labels:
        mask = y_true == k
        if mask.any():
            mean_prob[k] = y_prob[mask].mean(axis=0)
    pd.DataFrame(mean_prob, index=names, columns=names).to_csv(
        out_dir / f"{split_name}_softmax_mean.csv"
    )
    _heatmap(
        mean_prob,
        names,
        out_dir / f"{split_name}_softmax_mean.png",
        f"{split_name} mean softmax by true class",
        ".2f",
        0.0,
        1.0,
    )
    LOGGER.info("Wrote %s y_true/y_pred/y_prob and plots → %s", split_name, out_dir)


def run(config: TrainConfig, out_dir: Path, device: torch.device) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_window_splits(config)
    for name, (x, y, z) in splits.items():
        LOGGER.info("%s windows: %s  labels: %s", name, x.shape, np.bincount(y))

    LOGGER.info("Encoder=%s  features_len=%d", config.encoder, config.features_len)
    model = build_encoder(config).to(device)
    temporal = TC(config, device).to(device)

    train_dl, valid_dl, test_dl = make_loaders(splits, config, "self_supervised")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("Self-supervised pretraining")
    Trainer(model,temporal,model_opt, temporal_opt,train_dl,valid_dl,device,config,"self_supervised",config.num_epoch)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict": temporal.state_dict(),
        "config": {
            "encoder": config.encoder,
            "window": config.window,
            "stride_windows": config.stride_windows,
            "features_len": config.features_len,
            "num_classes": config.num_classes,
        },
    }
    ckpt_path = out_dir / "ckp_last.pt"
    torch.save(checkpoint, ckpt_path)
    LOGGER.info("Wrote encoder checkpoint → %s", ckpt_path)

    _maybe_freeze_for_linear(model, config)
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "train_linear")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    Trainer(model,temporal,model_opt,temporal_opt,train_dl,valid_dl,device,config,"train_linear",config.linear_epoch)

    _, _, val_pred, val_true, val_prob = model_evaluate(
        model, temporal, valid_dl, device, "train_linear"
    )
    _, _, test_pred, test_true, test_prob = model_evaluate(
        model, temporal, test_dl, device, "train_linear"
    )
    report = (
        _experiment_header(config, "ts-tcc")
        + _report("val", val_true, val_pred)
        + "\n"
        + _report("test", test_true, test_pred)
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    _save_eval(out_dir, "val", val_true, val_pred, val_prob)
    _save_eval(out_dir, "test", test_true, test_pred, test_prob)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "linear_head.pt")
    LOGGER.info("Wrote report → %s", report_path)


def run_ca_tcc(config: TrainConfig, out_dir: Path, device: torch.device, label_fraction: float = 0.01) -> None:
    """CA-TCC: TS-TCC pretrain, fine-tune on 1% labels, pseudo-labels, SupCon, linear eval."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_window_splits(config)
    x_train, y_train = splits["train"][:2]
    few_idx = stratified_indices(y_train, label_fraction, config.seed)
    few_xy = (x_train[few_idx], y_train[few_idx])
    LOGGER.info("Labeled fraction %.0f%%: %d / %d windows", label_fraction * 100, len(few_idx), len(y_train))

    LOGGER.info("Encoder=%s  features_len=%d", config.encoder, config.features_len)
    model = build_encoder(config).to(device)
    temporal = TC(config, device).to(device)

    #CA-TCC Stage 1: Pretrain self-supervised on all training windows
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "self_supervised")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 1: self-supervised pretraining")
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "self_supervised", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ssl.pt")

    #CA-TCC Stage 2: Finetune supervised with fractions of windows with labels
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "ft_1p", train_xy=few_xy, batch_size=16)
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 2: fine-tune on %.0f%% labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "ft_1p", config.linear_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ft.pt")

    #CA-TCC Stage 3: Use finetuned model to pseudo labels the rest of the training windows
    y_pseudo = pseudo_labels(model, x_train, device, config.batch_size)
    agree = float((y_pseudo == y_train).mean())
    LOGGER.info("CA-TCC stage 3: pseudo labels agree with true labels on %.3f of train", agree)
    np.save(out_dir / "pseudo_labels.npy", y_pseudo)

    #CA-TCC Stage 4: Continue finetune constrative learning with pseudo labels 
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "SupCon", train_xy=(x_train, y_pseudo)
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 4: SupCon on pseudo labels")
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "SupCon", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_supcon.pt")

    # CA-TCC stage 5: CNN and DINOv3 freeze down to the head unless --finetune-encoder
    _maybe_freeze_for_linear(model, config)
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "train_linear", train_xy=few_xy, batch_size=16
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC stage 5: head on %.0f%% true labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "train_linear", config.linear_epoch)

    _, _, val_pred, val_true, val_prob = model_evaluate(
        model, temporal, valid_dl, device, "train_linear"
    )
    _, _, test_pred, test_true, test_prob = model_evaluate(
        model, temporal, test_dl, device, "train_linear"
    )
    report = (
        _experiment_header(
            config,
            "ca-tcc",
            extra=f"label_fraction={label_fraction} n_labeled={len(few_idx)} pseudo_agree={agree:.4f}",
        )
        + _report("val", val_true, val_pred)
        + "\n"
        + _report("test", test_true, test_pred)
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    _save_eval(out_dir, "val", val_true, val_pred, val_prob)
    _save_eval(out_dir, "test", test_true, test_pred, test_prob)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "linear_head.pt")
    LOGGER.info("Wrote report → %s", report_path)
