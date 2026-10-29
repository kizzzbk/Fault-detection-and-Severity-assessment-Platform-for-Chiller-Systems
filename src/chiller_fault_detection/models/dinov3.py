"""DINOv3 encoder with the same TS-TCC contract as ``base_Model``.

A sensor window ``(B, C, T)`` is drawn as one grayscale image (channels along
height, time along width), resized, repeated to 3 channels, and ImageNet-
normalized. The pretrained ViT returns patch tokens. Those tokens are projected
to ``(B, final_out_channels, N)`` so the temporal-contrasting module can keep
its current shape. A linear layer named ``logits`` maps the same map to classes.

``N`` is the number of ViT patches, ``(img_size / patch_size) ** 2``. It is not
the CNN length 18. The backbone stays frozen unless ``--finetune-encoder``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Short names used by ``--dinov3-variant``. All are patch-16 ViTs.
HF_MODEL_IDS: dict[str, str] = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# Match the Hugging Face DINOv3 image processor.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def n_dino_tokens(img_size: int, patch_size: int) -> int:
    """Patch tokens a ViT emits for a square image. This is ``features_len``."""
    if patch_size < 1 or img_size < patch_size or img_size % patch_size != 0:
        raise ValueError(
            f"img_size {img_size} must be a positive multiple of patch_size {patch_size}"
        )
    grid = img_size // patch_size
    return grid * grid


def load_dinov3_backbone(variant: str) -> nn.Module:
    """Load a pretrained DINOv3 ViT from Hugging Face.

    The first call downloads the checkpoint. Meta's repos are gated: the
    license has to be accepted on the model page, then ``huggingface-cli login``.
    """
    if variant not in HF_MODEL_IDS:
        known = ", ".join(HF_MODEL_IDS)
        raise ValueError(f"Unknown DINOv3 variant {variant!r}; use one of: {known}")
    hf_id = HF_MODEL_IDS[variant]
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv3 needs the 'transformers' package. Install project dependencies and retry."
        ) from exc
    try:
        return AutoModel.from_pretrained(hf_id)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DINOv3 weights for {variant!r} ({hf_id}). "
            "Accept the model license on Hugging Face and run `huggingface-cli login`."
        ) from exc


class Dinov3Encoder(nn.Module):
    """Frozen (by default) DINOv3 ViT, a channel projection, and a class head.

    ``forward`` returns ``(logits, features)`` with ``features`` shaped
    ``(B, final_out_channels, N)``.
    """

    def __init__(self, configs, backbone: nn.Module | None = None):
        super().__init__()
        d3 = configs.dinov3
        self.img_size = d3.img_size
        self.patch_size = d3.patch_size
        self.n_tokens = n_dino_tokens(d3.img_size, d3.patch_size)
        if configs.features_len != self.n_tokens:
            raise ValueError(
                f"TrainConfig.features_len is {configs.features_len}, "
                f"but img_size={d3.img_size} and patch_size={d3.patch_size} "
                f"produce {self.n_tokens} tokens"
            )

        self.backbone = backbone if backbone is not None else load_dinov3_backbone(d3.variant)
        hidden = int(self.backbone.config.hidden_size)
        model_patch = int(self.backbone.config.patch_size)
        if model_patch != d3.patch_size:
            raise ValueError(
                f"Backbone patch size is {model_patch}, config asked for {d3.patch_size}"
            )

        if hidden == configs.final_out_channels:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
        else:
            self.feature_proj = nn.Sequential(
                nn.Conv1d(hidden, configs.final_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(configs.final_out_channels),
                nn.GELU(),
            )

        # Name contains "logits" so the linear-probe freeze keeps only this layer.
        self.logits = nn.Linear(
            self.n_tokens * configs.final_out_channels, configs.num_classes
        )
        self._backbone_frozen = False
        self.register_buffer(
            "pixel_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def freeze_backbone(self) -> None:
        """Stop gradients into the ViT and keep it in eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def train(self, mode: bool = True):
        """Trainer calls ``model.train()`` every fit. A frozen ViT stays in eval."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def _windows_to_images(self, x_in: torch.Tensor) -> torch.Tensor:
        """Rasterize ``(B, C, T)`` into an ImageNet-normalized ``(B, 3, H, W)``."""
        if x_in.ndim != 3:
            raise ValueError(f"Expected windows shaped (B, C, T), got {tuple(x_in.shape)}")
        flat = x_in.flatten(1)
        low = flat.min(dim=1).values.view(-1, 1, 1)
        high = flat.max(dim=1).values.view(-1, 1, 1)
        scaled = (x_in - low) / (high - low).clamp_min(1e-6)
        image = scaled.unsqueeze(1)
        image = F.interpolate(
            image,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image = image.repeat(1, 3, 1, 1)
        return (image - self.pixel_mean) / self.pixel_std

    def _hidden(self, images: torch.Tensor) -> torch.Tensor:
        if self._backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)
        return outputs.last_hidden_state

    def apply_patch_mask(
        self, images: torch.Tensor, mask_ratio: float = 0.4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized 16x16 patch masking directly on (B, 3, H, W) images.

        Returns:
            masked_images: (B, 3, H, W) image tensor with selected patches zeroed out.
            mask: (B, N) boolean tensor where True indicates masked patches.
        """
        b, c, h, w = images.shape
        p = self.patch_size
        gh = h // p
        gw = w // p
        n_tokens = gh * gw

        # Sample random boolean mask of shape (B, n_tokens)
        probs = torch.rand(b, n_tokens, device=images.device)
        mask = probs < mask_ratio

        # Reshape images into non-overlapping (B, n_tokens, C, P, P) patches
        patches = (
            images.view(b, c, gh, p, gw, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, n_tokens, c, p, p)
        )
        # Zero out the selected patches
        patches = patches.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)

        # Invert permutation and reshape back to (B, C, H, W)
        masked_images = (
            patches.view(b, gh, gw, c, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, h, w)
        )
        return masked_images, mask

    def forward(self, x_in: torch.Tensor, mask_ratio: float = 0.0):
        images = self._windows_to_images(x_in)
        mask = None
        if mask_ratio > 0.0 and self.training:
            images, mask = self.apply_patch_mask(images, mask_ratio=mask_ratio)

        hidden = self._hidden(images)
        n_prefix = 1 + int(self.backbone.config.num_register_tokens)
        patches = hidden[:, n_prefix:, :]
        if patches.shape[1] != self.n_tokens:
            raise RuntimeError(
                f"DINOv3 returned {patches.shape[1]} patch tokens, expected {self.n_tokens}"
            )
        features = self.feature_proj(patches.transpose(1, 2))
        logits = self.logits(features.reshape(features.shape[0], -1))
        if mask is not None:
            return logits, features, mask
        return logits, features

