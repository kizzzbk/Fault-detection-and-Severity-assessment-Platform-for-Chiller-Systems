"""Weak and strong views for TS-TCC, built once before training.

Weak view: jitter then scale (Section 3.1).
Strong view: permute segments, then jitter.
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
