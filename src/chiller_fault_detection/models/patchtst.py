"""PatchTST encoder with the same TS-TCC contract as ``base_Model``.

CNN (``models.model.base_Model``) and this class are alternative encoders.
Downstream TC / losses stay unchanged if ``forward`` returns:

* ``logits``: ``(B, num_classes)``
* ``features``: ``(B, final_out_channels, N)`` with ``N = n_patches(T)``

Matches Tsinghua PatchTST (Nie et al., ICLR 2023) on the backbone:

* channel-independent patching and shared Transformer weights
* Post-LN + BatchNorm encoder layers (not PyTorch Pre-LN LayerNorm)
* classification head uses the last patch of every channel

No RevIN: TS-TCC jitter/scale would be cancelled by instance norm.
No channel-mixing 1x1 over ``C * d_model``. TC sees a mean over channels;
the classifier still sees all channels via the last-patch head.
"""

from __future__ import annotations

import torch
from torch import nn


def n_patches(length: int, patch_len: int, patch_stride: int) -> int:
    """Number of tokens ``unfold`` produces on a length-``length`` series."""
    if patch_len < 1 or patch_stride < 1:
        raise ValueError("patch_len and patch_stride must be >= 1")
    if length < patch_len:
        raise ValueError(f"window {length} is shorter than patch_len {patch_len}")
    return (length - patch_len) // patch_stride + 1


class _BatchNormSeq(nn.Module):
    """BatchNorm over ``d_model`` for sequences shaped ``(B, N, D)``."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class TSTEncoderLayer(nn.Module):
    """One Tsinghua-style encoder block: attention / FFN, Post-LN, BatchNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_attn = _BatchNormSeq(d_model)
        self.norm_ff = _BatchNormSeq(d_model)

    def forward(self, src):
        attn_out, _ = self.self_attn(src, src, src, need_weights=False)
        src = self.norm_attn(src + self.dropout_attn(attn_out))
        src = self.norm_ff(src + self.dropout_ff(self.ff(src)))
        return src


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST backbone + last-patch classification head.

    Input ``x_in`` is ``(B, C, T)``, same layout as the CNN encoder.
    """

    def __init__(self, configs):
        super().__init__()
        pt = configs.patchtst
        if pt.d_model % pt.n_heads != 0:
            raise ValueError(
                f"d_model {pt.d_model} must be divisible by n_heads {pt.n_heads}"
            )

        self.patch_len = pt.patch_len
        self.patch_stride = pt.patch_stride
        self.n_patches = n_patches(configs.window, pt.patch_len, pt.patch_stride)
        self.n_vars = configs.input_channels
        self.d_model = pt.d_model

        self.patch_embed = nn.Linear(pt.patch_len, pt.d_model)
        # Tsinghua default: pe='zeros', learn_pe=True.
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, pt.d_model))
        self.embed_dropout = nn.Dropout(pt.dropout)
        self.encoder = nn.ModuleList(
            [
                TSTEncoderLayer(pt.d_model, pt.n_heads, pt.d_ff, pt.dropout)
                for _ in range(pt.n_layers)
            ]
        )

        if pt.d_model == configs.final_out_channels:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm1d(pt.d_model),
                nn.GELU(),
            )
        else:
            self.feature_proj = nn.Sequential(
                nn.Conv1d(pt.d_model, configs.final_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(configs.final_out_channels),
                nn.GELU(),
            )

        # Tsinghua ClassificationHead: last patch × channels, then linear.
        self.logits = nn.Sequential(
            nn.Dropout(pt.head_dropout),
            nn.Linear(self.n_vars * pt.d_model, configs.num_classes),
        )

    def forward(self, x_in):
        # (B, C, T) → (B, C, N, P) → (B*C, N, P)
        patches = x_in.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        batch, channels, n_tok, _patch_len = patches.shape
        tokens = patches.reshape(batch * channels, n_tok, _patch_len)
        tokens = self.patch_embed(tokens) + self.pos_embedding[:, :n_tok, :]
        tokens = self.embed_dropout(tokens)
        for layer in self.encoder:
            tokens = layer(tokens)

        # (B*C, N, D) → (B, C, N, D)
        hidden = tokens.reshape(batch, channels, n_tok, -1)

        # TC contract: mean over channels keeps independence, then (B, D, N).
        features = self.feature_proj(hidden.mean(dim=1).transpose(1, 2))

        last_patch = hidden[:, :, -1, :].reshape(batch, -1)
        logits = self.logits(last_patch)
        return logits, features
