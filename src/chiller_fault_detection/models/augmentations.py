"""Weak and strong views for TS-TCC, built once before training.

Weak view: jitter then scale (Section 3.1).
CNN strong view: permute segments, then jitter.
PatchTST and DINOv3 strong views: mild jitter then mask time spans. Neither
permutes, because both add a position vector to each token and a shuffled
timeline would be read as a real order.
Input shape is ``(N, C, T)``. The same scale factor is shared across channels
so a multivariate window stays one sample. Permutation shuffles time for every
channel together.
"""

from __future__ import annotations

import numpy as np
import torch


def DataTransform(sample, config) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(weak, strong)`` views for a batch of windows.

    Called on the full training set before the training loop, matching the
    TS-TCC dataloader: augmentations are fixed views, not redrawn each step.
    """
    x = _as_numpy(sample)
    aug = config.augmentation
    weak = scaling(jitter(x, sigma=aug.weak_jitter_ratio), sigma=aug.jitter_scale_ratio)
    encoder = getattr(config, "encoder", "cnn")
    if encoder == "patchtst":
        pt = config.patchtst
        strong = mask_patches(
            jitter(x, sigma=aug.patchtst_jitter_ratio),
            patch_len=pt.patch_len,
            patch_stride=pt.patch_stride,
            mask_ratio=aug.patchtst_mask_ratio,
        )
    elif encoder == "dinov3":
        d3 = config.dinov3
        strong = mask_patches(
            jitter(x, sigma=aug.dinov3_jitter_ratio),
            patch_len=d3.mask_patch_len,
            patch_stride=d3.mask_patch_stride,
            mask_ratio=aug.dinov3_mask_ratio,
        )
    else:
        strong = jitter(
            permutation(x, max_segments=aug.max_seg),
            sigma=aug.jitter_ratio,
        )
    return _as_tensor(weak), _as_tensor(strong)


def jitter(x, sigma: float = 0.8):
    x = _as_numpy(x)
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)


def scaling(x, sigma: float = 1.1):
    """Multiply each window by a per-timestep factor drawn around 2.0."""
    x = _as_numpy(x)
    factor = np.random.normal(loc=2.0, scale=sigma, size=(x.shape[0], x.shape[2])).astype(
        np.float32
    )
    return x * factor[:, np.newaxis, :]


def mask_patches(x, patch_len: int, patch_stride: int, mask_ratio: float = 0.4):
    """Zero a random subset of patches, same idea as PatchTST masked pretraining."""
    x = _as_numpy(x)
    n_win, _, length = x.shape
    if patch_len < 1 or patch_stride < 1:
        raise ValueError("patch_len and patch_stride must be >= 1")
    if length < patch_len:
        raise ValueError(f"window {length} is shorter than patch_len {patch_len}")
    n_tok = (length - patch_len) // patch_stride + 1
    n_mask = min(n_tok, max(1, int(round(n_tok * mask_ratio))))
    out = x.copy()
    for i in range(n_win):
        chosen = np.random.choice(n_tok, size=n_mask, replace=False)
        for tok in chosen:
            start = int(tok) * patch_stride
            out[i, :, start : start + patch_len] = 0.0
    return out


def permutation(x, max_segments: int = 5, seg_mode: str = "random"):
    x = _as_numpy(x)
    length = x.shape[2]
    max_segments = max(int(max_segments), 2)
    n_seg_hi = min(max_segments, length)
    num_segs = np.random.randint(1, n_seg_hi, size=(x.shape[0],))

    ret = np.empty_like(x)
    steps = np.arange(length)
    for i, pat in enumerate(x):
        n_seg = int(num_segs[i])
        if n_seg <= 1:
            ret[i] = pat
            continue
        if seg_mode == "random":
            n_cuts = min(n_seg - 1, length - 2)
            if n_cuts < 1:
                ret[i] = pat
                continue
            split_points = np.sort(np.random.choice(length - 2, n_cuts, replace=False)) + 1
            splits = np.split(steps, split_points)
        else:
            splits = np.array_split(steps, n_seg)
        order = np.random.permutation(len(splits))
        warp = np.concatenate([splits[j] for j in order])
        ret[i] = pat[:, warp]
    return ret


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected windows shaped (N, C, T), got {arr.shape}")
    return arr


def _as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
