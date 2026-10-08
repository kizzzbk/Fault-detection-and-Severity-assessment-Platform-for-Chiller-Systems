"""PatchTST encoder with the same TS-TCC contract as ``base_Model``.

CNN (``models.model.base_Model``) and this class are alternative encoders.
Downstream TC / losses stay unchanged if ``forward`` returns:

* ``logits``: ``(B, num_classes)``
* ``features``: ``(B, final_out_channels, N)`` with ``N = n_patches(T)``

Inside: channel-independent patching → Transformer → 1x1 conv mixes
channels down to ``final_out_channels`` (default 128). No RevIN, so the
existing weak/strong time-domain augmentations still reach the encoder.
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


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST backbone + linear probe head.

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

        self.patch_embed = nn.Linear(pt.patch_len, pt.d_model)
        self.pos_embedding = nn.Parameter(torch.empty(1, self.n_patches, pt.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(pt.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pt.d_model,
            nhead=pt.n_heads,
            dim_feedforward=pt.d_ff,
            dropout=pt.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=pt.n_layers,
            enable_nested_tensor=False,
        )

        mixed_dim = configs.input_channels * pt.d_model
        self.channel_mix = nn.Sequential(
            nn.Conv1d(mixed_dim, configs.final_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(configs.final_out_channels),
        )
        self.logits = nn.Linear(
            configs.features_len * configs.final_out_channels,
            configs.num_classes,
        )

    def forward(self, x_in):
        # (B, C, T) → (B, C, N, P) → (B*C, N, P)
        patches = x_in.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        batch, channels, n_patches, patch_len = patches.shape
        tokens = patches.reshape(batch * channels, n_patches, patch_len)
        tokens = self.patch_embed(tokens) + self.pos_embedding[:, :n_patches, :]
        tokens = self.embed_dropout(tokens)
        tokens = self.transformer(tokens)

        # (B*C, N, D) → (B, C*D, N) → (B, C_out, N)
        hidden = tokens.reshape(batch, channels, n_patches, -1)
        hidden = hidden.permute(0, 1, 3, 2).reshape(batch, -1, n_patches)
        features = self.channel_mix(hidden)

        logits = self.logits(features.reshape(batch, -1))
        return logits, features
