"""CA-TCC with a FixMatch-style confidence threshold on pseudo labels.

Same five stages as ``trainer.run_ca_tcc``. Only stage 3 and the input of
stage 4 differ:

* stage 3 keeps a pseudo label only when ``max softmax > tau``
* the labeled windows always keep their true label
* stage 4 SupCon trains on the kept subset only

The threshold is applied once, offline, between stage 2 and stage 4. It is not
the online two-batch FixMatch training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chiller_fault_detection.models.TC import TC
from chiller_fault_detection.models.trainer import (
    Trainer,
    TrainConfig,
    _experiment_header,
    _maybe_freeze_for_linear,
    _optimizers,
    _report,
    _save_eval,
    build_encoder,
    load_window_splits,
    make_loaders,
    model_evaluate,
    stratified_indices,
)

LOGGER = logging.getLogger("ts_tcc")


@dataclass
class FixMatchConfig:
    tau: float = 0.80


def pseudo_labels_with_confidence(
    model, x: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pred, conf)``: argmax class and its softmax probability per window."""
    model.eval()
    preds = []
    confs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits, _ = model(batch)
            conf, pred = F.softmax(logits, dim=1).max(dim=1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def fixmatch_mask(conf: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask, same length as ``conf``: keep windows with confidence above ``tau``."""
    return conf > tau


def run_ca_tcc_fixmatch(
    config: TrainConfig,
    fixmatch: FixMatchConfig,
    out_dir: Path,
    device: torch.device,
    label_fraction: float = 0.01,
) -> None:
    """CA-TCC where stage 4 only sees pseudo labels above a fixed confidence ``tau``."""
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

    # Stage 1: self-supervised pretraining on all training windows
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "self_supervised")
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC-FixMatch stage 1: self-supervised pretraining")
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "self_supervised", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ssl.pt")

    # Stage 2: supervised fine-tune on the labeled fraction
    train_dl, valid_dl, test_dl = make_loaders(splits, config, "ft_1p", train_xy=few_xy, batch_size=16)
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC-FixMatch stage 2: fine-tune on %.0f%% labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "ft_1p", config.linear_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_ft.pt")

    # Stage 3: pseudo label every window, keep only confident ones
    y_pseudo, conf = pseudo_labels_with_confidence(model, x_train, device, config.batch_size)
    agree_all = float((y_pseudo == y_train).mean())

    keep = fixmatch_mask(conf, fixmatch.tau)
    y_pseudo[few_idx] = y_train[few_idx]
    keep[few_idx] = True

    n_kept = int(keep.sum())
    keep_rate = float(keep.mean())
    agree_kept = float((y_pseudo[keep] == y_train[keep]).mean())
    kept_per_class = np.bincount(y_pseudo[keep], minlength=config.num_classes)
    LOGGER.info(
        "CA-TCC-FixMatch stage 3: tau=%.2f agree_all=%.3f kept=%d/%d (%.3f) agree_kept=%.3f",
        fixmatch.tau,
        agree_all,
        n_kept,
        len(keep),
        keep_rate,
        agree_kept,
    )
    LOGGER.info("CA-TCC-FixMatch stage 3: kept per class %s", kept_per_class.tolist())
    if n_kept < config.batch_size:
        LOGGER.warning(
            "Only %d windows kept (< batch_size %d); tau=%.2f may be too strict",
            n_kept,
            config.batch_size,
            fixmatch.tau,
        )
    np.save(out_dir / "pseudo_labels.npy", y_pseudo)
    np.save(out_dir / "pseudo_conf.npy", conf)
    np.save(out_dir / "pseudo_keep.npy", keep)

    # Stage 4: SupCon on the kept pseudo-labeled subset
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "SupCon", train_xy=(x_train[keep], y_pseudo[keep])
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC-FixMatch stage 4: SupCon on %d kept pseudo labels", n_kept)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "SupCon", config.num_epoch)
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "ckp_supcon.pt")

    # Stage 5: head on the labeled fraction
    _maybe_freeze_for_linear(model, config)
    train_dl, valid_dl, test_dl = make_loaders(
        splits, config, "train_linear", train_xy=few_xy, batch_size=16
    )
    model_opt, temporal_opt = _optimizers(model, temporal, config)
    LOGGER.info("CA-TCC-FixMatch stage 5: head on %.0f%% true labels", label_fraction * 100)
    Trainer(model, temporal, model_opt, temporal_opt, train_dl, valid_dl, device, config, "train_linear", config.linear_epoch)

    _, _, val_pred, val_true, val_prob = model_evaluate(
        model, temporal, valid_dl, device, "train_linear"
    )
    _, _, test_pred, test_true, test_prob = model_evaluate(
        model, temporal, test_dl, device, "train_linear"
    )
    extra = (
        f"label_fraction={label_fraction} n_labeled={len(few_idx)} tau={fixmatch.tau}\n"
        f"pseudo_agree_all={agree_all:.4f} kept={n_kept}/{len(keep)} "
        f"keep_rate={keep_rate:.4f} pseudo_agree_kept={agree_kept:.4f}\n"
        f"kept_per_class={kept_per_class.tolist()}"
    )
    report = (
        _experiment_header(config, "ca-tcc-fixmatch", extra=extra)
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
