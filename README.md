# DINOv3 Integration with TS-TCC / CA-TCC

This guide documents the encoder swap from the original TS-TCC CNN
(`base_Model`) to a pretrained DINOv3 ViT. The training entrypoint, loss
functions, temporal-contrasting module, and windowed RP-1043 splits are
unchanged. The supported task is 8-way `fault_id` classification on windows
of shape `(B, C, T)`.

`--encoder cnn` and `--encoder patchtst` keep their previous behaviour.
`--encoder dinov3` selects `Dinov3Encoder`.

CNN and PatchTST train from random initialisation. DINOv3 does not: the ViT
weights come from Meta’s public checkpoint and stay frozen unless
`--finetune-encoder` is set. Full training and a held-out comparison against
CNN have not been recorded in this repository.

## 1. Architecture

```text
Original CNN (base_Model):
Windows (B, C, T) → 3 conv blocks → features (B, 128, 18)
                  → Linear logits (B, 8)
                  → TC reads features

Dinov3Encoder:
Windows (B, C, T) → image adapter → DINOv3 → projection
                  → features (B, 128, N)
                  → Linear logits (B, 8)
                  → TC reads features
```

`N = (img_size / patch_size)²`. The default `224 / 16` grid is 14×14, so
`N = 196`. CNN length 18 is the temporal size after three conv/pool blocks
on `T = 128`; it is not reused.

| Component | CNN (`base_Model`) | DINOv3 (`Dinov3Encoder`) |
| --- | --- | --- |
| Backbone | Three 1-D conv blocks, trained from scratch | Meta DINOv3 ViT, loaded from Hugging Face |
| Backbone gradients | Updated until the linear probe | Frozen in `eval()` + `torch.no_grad()` unless `--finetune-encoder` |
| Projection | None (conv already emits 128 channels) | `Conv1d` 1×1 + BatchNorm + GELU, `hidden → 128`, trainable |
| Classification head | `Linear` named `logits` | Same contract, `N × 128 → 8` |
| Temporal contrasting (`TC`) | Original module, trainable | Same module, trainable |
| Loss | Original TS-TCC / CA-TCC losses | Same formulas and coefficients |
| `features` contract | `(B, 128, N)` with `N > TC.timesteps` | Same contract; only `N` changes |

Freezing means DINOv3 weights are not updated. `build_encoder` calls
`freeze_backbone()` before stage 1. The ViT stays in `eval()` even when
`Trainer` calls `model.train()`, so DropPath does not resample a frozen
backbone. The projection, head, and `TC` receive gradients. Adam is built
with `filter(lambda p: p.requires_grad, ...)`.

`Dinov3Encoder.forward` returns the same pair as `base_Model`:
`(logits, features)`. `TC.py` therefore does not need a DINOv3 branch.
`TrainConfig` rejects any `img_size` whose token count is `≤ TC.timesteps`
(default 6). With patch size 16 the smallest usable image is 64 (`N = 16`).

### Input adaptation

DINOv3 expects an ImageNet-normalised RGB image. The public preprocess still
writes residualised windows `(B, C, T)`; `C` is `len(MODEL_FEATURE_COLUMNS)`
and `T` defaults to 128. The adapter in
`Dinov3Encoder._windows_to_images` is the only place a window becomes an
image:

1. Min–max scale each window independently to `[0, 1]`.
2. Treat channels as image height and time as width: `(B, 1, C, T)`.
3. Resize bilinearly to `(img_size, img_size)` (default 224×224).
4. Repeat the single plane to three channels.
5. Subtract ImageNet mean `(0.485, 0.456, 0.406)` and divide by
   `(0.229, 0.224, 0.225)`, matching the Hugging Face DINOv3 processor.
6. Run the ViT. Drop the CLS token and register tokens; keep the `N` patch
   tokens.
7. Project token width `hidden` (384 for `vits16`) to
   `final_out_channels` (128). Flatten the same map into `logits`.

The sensor layout is not a photograph. Resize interpolates the `(C, T)`
plane; it does not add measurements. Labels stay on windows. `logits` remain
`(B, 8)` so the original cross-entropy path is unchanged.

### Strong-view augmentation

The weak view is still jitter then scale, for every encoder.

The CNN strong view permutes time segments and then applies a large jitter
(`σ = 0.8`). DINOv3 adds a fixed positional vector to each patch, so a
shuffled timeline would be read as a real order and the contrastive loss
would pull two different orders together. The DINOv3 strong view therefore
follows the PatchTST policy: mild jitter (`σ = 0.2`) then mask about 40% of
time spans (`mask_patch_len = 8`, `mask_patch_stride = 8`). Coefficients live
in `AugmentConfig` (`dinov3_jitter_ratio`, `dinov3_mask_ratio`) and are not
shared with the CNN permute view.

## 2. Code changes and rationale

| File | Change | Reason |
| --- | --- | --- |
| [`src/chiller_fault_detection/models/dinov3.py`](src/chiller_fault_detection/models/dinov3.py) | Adapter, backbone loader, projection, head | Embed DINOv3 while keeping `(logits, features)` |
| [`src/chiller_fault_detection/models/trainer.py`](src/chiller_fault_detection/models/trainer.py) | `Dinov3Config`, `features_len`, `build_encoder`, freeze, `_encoder_lr` | Select the encoder and stop ViT updates by default |
| [`src/chiller_fault_detection/models/augmentations.py`](src/chiller_fault_detection/models/augmentations.py) | `encoder == "dinov3"` strong view | Avoid permuting a position-sensitive token sequence |
| [`scripts/train.py`](scripts/train.py) | `--encoder dinov3`, `--dinov3-variant`, `--img-size`, default batch 16 | Expose the swap without a second entrypoint |
| [`pyproject.toml`](pyproject.toml) | `transformers>=4.56` | `AutoModel.from_pretrained` is required to read the checkpoint |
| [`src/chiller_fault_detection/models/model.py`](src/chiller_fault_detection/models/model.py) | Comment only | CNN file stays the conv `base_Model` |
| [`src/chiller_fault_detection/models/fixmatch.py`](src/chiller_fault_detection/models/fixmatch.py) | Comment only | FixMatch already calls `build_encoder` / `_maybe_freeze_for_linear` |

`TC.py`, `loss.py`, and `attention.py` are unchanged. They only read
`features` as `(B, 128, N)`. The public loaders and residualiser are
unchanged: windows on disk stay `(B, C, T)`.

Artifact directories remain `{method}_{encoder}` (plus `_ft` when
`--finetune-encoder` is set), so a DINOv3 run does not overwrite CNN
weights.

## 3. Environment and checkpoints

The default backbone is Meta’s
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
(`vits16`, hidden size 384). The ViT was not trained in this project.

`--dinov3-variant` maps to the following Hugging Face repos. Only `vits16`
is the intended default; the larger IDs are accepted by the CLI but have
not been run here.

| `--dinov3-variant` | Hugging Face repo |
| --- | --- |
| `vits16` | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| `vits16plus` | `facebook/dinov3-vits16plus-pretrain-lvd1689m` |
| `vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `vitl16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` |

The repos are gated. Accept the license on the model page, then
`huggingface-cli login`. `load_dinov3_backbone` calls
`AutoModel.from_pretrained(hf_id)`. The first call downloads into the
Hugging Face cache (`~/.cache/huggingface/hub/`). The git repository does
not contain ViT weights, and the loader does not read a project-local
`.pt` / `.pth` file.

From the repository root:

```bash
uv sync
huggingface-cli login
```

Python is `>=3.13` ([`pyproject.toml`](pyproject.toml)). `scripts/train.py`
uses CUDA when a device with compute capability ≥ 7.5 is present; otherwise
it falls back to CPU. Pass `--device cuda` or `--device cpu` to override.

## 4. Data and training command

Training reads the existing processed splits under
`artifacts/processed/splits/`. Each example is one window of residualised
channels plus a `fault_id` in `{0,…,7}`. Default window / stride are 128 /
64, matching the CNN path.

Run all commands from the repository root.

Default (frozen ViT, CA-TCC, 1% labels at the fine-tune stage):

```bash
uv run python scripts/train.py --encoder dinov3 --method ca-tcc \
  --dinov3-variant vits16 --img-size 224
```

Unfreeze the ViT from stage 1 (learning rate capped at `1e-5`):

```bash
uv run python scripts/train.py --encoder dinov3 --finetune-encoder \
  --dinov3-variant vits16
```

Smaller images if memory is tight. `img_size` must be a multiple of 16 and
`N` must exceed `TC.timesteps`:

```bash
uv run python scripts/train.py --encoder dinov3 --img-size 64 --batch-size 32
```

`--batch-size` omitted: CNN and PatchTST use 128; DINOv3 uses 16. A written
`--batch-size` is kept as-is. Outputs go to
`artifacts/{method}_{encoder}[/ _ft]`.

`--dinov3-variant` and `--img-size` are ignored unless `--encoder dinov3`.

## 5. Loss and parameter updates

The loss code path is the original `model_train`. Replacing the encoder
changes how `features` and `logits` are produced, not the objectives.

Let \(f_1, f_2\) be the two augmented `features` tensors after L2
normalisation on the channel axis. `TC` returns a temporal-contrasting term
and a projected vector \(z\) (or \(h\) in SupCon). Temperature for NT-Xent
is \(\tau = 0.2\). \(K = 6\) future steps.

**Stage 1 (self-supervised, TS-TCC and CA-TCC):**

\[
\mathcal{L}_{\mathrm{ssl}}
= \mathcal{L}_{\mathrm{tc}}(f_1,f_2)+\mathcal{L}_{\mathrm{tc}}(f_2,f_1)
+ 0.7\,\mathcal{L}_{\mathrm{NT\text{-}Xent}}(z_1,z_2)
\]

**CA-TCC stage 4 (SupCon on pseudo-labels):**

\[
\mathcal{L}_{\mathrm{sup}}
= 0.01\bigl(\mathcal{L}_{\mathrm{tc}}(f_1,f_2)+\mathcal{L}_{\mathrm{tc}}(f_2,f_1)\bigr)
+ 0.1\,\mathcal{L}_{\mathrm{SupCon}}(h_1,h_2,\hat{y})
\]

**Labelled stages** (`ft_1p`, `train_linear`):

\[
\mathcal{L}_{\mathrm{ce}}=\mathrm{CrossEntropy}(\mathrm{logits}(x),y)
\]

`TC` is not stepped on labelled stages. `ReduceLROnPlateau` is applied only
there.

Write \(\theta_{\mathrm{ViT}}\), \(\theta_{\mathrm{proj}}\),
\(\theta_{\mathrm{head}}\), \(\theta_{\mathrm{TC}}\) for the backbone,
`feature_proj`, `logits`, and temporal module. The update is ordinary Adam
on parameters with `requires_grad`:

\[
\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}.
\]

| Stage | CNN updates | DINOv3 updates (default) |
| --- | --- | --- |
| 1 self-supervised | conv + \(\theta_{\mathrm{TC}}\) | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{TC}}\) |
| 2 CA-TCC `ft_1p` | conv + head | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{head}}\) |
| 4 SupCon | conv + \(\theta_{\mathrm{TC}}\) | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{TC}}\) |
| 5 linear probe | head only (`logits` in the name) | head only; projection is frozen in `_maybe_freeze_for_linear` |

Default DINOv3 keeps \(\nabla_{\theta_{\mathrm{ViT}}}\mathcal{L}=0\) on
every stage. Encoder and `TC` share one learning rate (no param groups):
\(3\cdot 10^{-4}\) when the ViT is frozen, \(10^{-5}\) when
`--finetune-encoder` is set. In the latter case stage 5 does not freeze
again.

## 6. Status and limitations

| Check | Status |
| --- | --- |
| Encoder contract `(logits, features)` | Implemented in `Dinov3Encoder` |
| Frozen-backbone default | Implemented in `build_encoder` / `freeze_backbone` |
| Shared `TC` and loss coefficients | Unchanged files; DINOv3 must emit 128-channel features |
| Dedicated smoke script | Not present |
| Full CA-TCC / TS-TCC run with `--encoder dinov3` | Not recorded in this repository |
| Held-out comparison with CNN or PatchTST | Not recorded |
| `--finetune-encoder` and variants other than `vits16` | Not recorded |

Report accuracy / F1 on the existing val and test splits after a full run.
Do not treat the CNN demo checkpoint `demo/models/ckp_ft.pt` as a DINOv3
weight file.

The raster adapter uses every residual channel as a single grayscale plane
copied to RGB. It does not encode physical units as colour, and bilinear
resize does not recover a higher sampling rate.

`--img-size` 32 yields `N = 4` and is rejected. Batch 128 on a 224² ViT is
likely to exhaust memory; the CLI default of 16 exists for that reason.

## 7. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Could not load DINOv3 weights` | Accept the Hugging Face license and run `huggingface-cli login` |
| `transformers` ImportError | `uv sync` so `transformers>=4.56` is installed |
| `features_len` / token-count error | `img_size` must be a positive multiple of 16 and `N > 6` |
| `Unknown encoder 'dinov3'` | Use the current `scripts/train.py`; older copies only listed `cnn` / `patchtst` |
| Out of memory | `--img-size 64` and a smaller `--batch-size`; stay on `vits16` |
| Backbone still in train mode | Default path must call `freeze_backbone()`; `Dinov3Encoder.train()` keeps the ViT in `eval()` |
| Artifact overwrites a CNN run | Confirm the output directory is `artifacts/{method}_dinov3`, not `*_cnn` |

## References

- [TS-TCC](https://arxiv.org/abs/2106.14112), [CA-TCC](https://arxiv.org/abs/2203.14244).
- [DINOv3](https://ai.meta.com/dinov3/), [DINOv3 in Transformers](https://huggingface.co/docs/transformers/model_doc/dinov3).
- [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
- ASHRAE RP-1043 chiller fault data, processed by
  [`src/chiller_fault_detection/data/public/`](src/chiller_fault_detection/data/public/).
