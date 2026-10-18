# Thay encoder bằng DINOv3

CA-TCC / TS-TCC giữ nguyên vòng train, loss, module temporal contrasting (`TC`) và dữ liệu cửa sổ `(B, C, T)`. Phần đổi là khối encoder: CNN (`base_Model`) được thay bằng ViT DINOv3 đã pretrain trên ảnh, cộng một lớp chiếu để `features` vẫn đúng contract mà `TC` đang đọc.

`--encoder cnn` và `--encoder patchtst` không đổi. `--encoder dinov3` mới gọi `Dinov3Encoder`.

## Pretrained checkpoint

Trọng số ViT **không được train trong bài này**. Chúng là checkpoint DINOv3 của Meta, pretrain trên LVD-1689M, tải qua Hugging Face:

| `--dinov3-variant` | Repo Hugging Face | Embedding |
| --- | --- | --- |
| `vits16` (mặc định) | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 |
| `vits16plus` | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | — |
| `vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 |
| `vitl16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | — |

`load_dinov3_backbone` trong [`src/chiller_fault_detection/models/dinov3.py`](src/chiller_fault_detection/models/dinov3.py) gọi `transformers.AutoModel.from_pretrained`. Lần đầu checkpoint được ghi vào cache Hugging Face trên máy (`~/.cache/huggingface/hub/`). Repo không chứa file `.pt` của ViT; CNN trong bài thì train từ đầu, không có bước tải này.

Repo Meta đang gated: chấp nhận license trên trang model, rồi `huggingface-cli login`. Thiếu gói `transformers>=4.56` hoặc chưa login thì hàm báo đúng việc đó.

CNN không có pretrained checkpoint. Mọi conv trong `base_Model` khởi tạo ngẫu nhiên và học trên cửa sổ RP-1043.

## Nhúng DINOv3 vào pipeline

DINOv3 nhận ảnh 3 kênh, không nhận cảm biến. Encoder mới là lớp bọc: vẽ cửa sổ thành ảnh, chạy ViT, bỏ token không dùng, chiếu về đúng shape `TC` đang chờ.

```
cửa sổ x ∈ R^{B×C×T}          C = 19, T = 128  (giống CNN)
        │
        ▼
  rasterize + resize 224×224 + lặp 3 kênh + chuẩn hóa ImageNet
        │
        ▼
  DINOv3 ViT  (pretrained; đóng gradient nếu không --finetune-encoder)
        │
        ▼
  bỏ CLS và register token, giữ N patch token
        │
        ▼
  feature_proj: Conv1d 1×1 + BN + GELU     384 → 128   (vits16)
        │
        ├──────── features ∈ R^{B×128×N}   ──► module TC (không sửa)
        │
        └──────── flatten → Linear tên logits ──► 8 lớp fault_id
```

`forward(x)` vẫn trả `(logits, features)` như CNN. `logits` có shape `(B, 8)`. `features` có shape `(B, 128, N)`:

- CNN: `N = 18` (ba khối conv + pool trên `T = 128`)
- DINOv3: `N = (img_size / patch_size)²`. Mặc định `224 / 16` → lưới 14×14 → `N = 196`

`N` phải lớn hơn `TC.timesteps` (mặc định 6). Ảnh cạnh 32 chỉ ra 4 token và bị `TrainConfig` từ chối; cạnh nhỏ nhất dùng được là 64 (`N = 16`).

Việc vẽ ảnh nằm trong `Dinov3Encoder._windows_to_images`: co từng cửa sổ về `[0, 1]`, coi 19 kênh là chiều cao và 128 bước là chiều rộng, phóng bilinear lên ảnh vuông, lặp thành RGB, trừ mean / chia std ImageNet. 19 cột cảm biến không được coi là ảnh RGB.

`TC.py`, `loss.py` và `attention.py` không sửa. Chúng chỉ thấy `features` dạng `(B, 128, N)`. Chiếu token về 128 kênh là để giữ được điều đó. `data/load.py` và `data/preprocess.py` cũng không sửa: cửa sổ trên đĩa vẫn `(B, C, T)`.

## Loss

Loss **không đổi khi thay encoder**. Cùng `model_train` trong [`trainer.py`](src/chiller_fault_detection/models/trainer.py), cùng `NTXentLoss` / `SupConLoss`, cùng hệ số 0.7 / 0.01 / 0.1. Encoder chỉ đổi cách tính `features` (và `logits` ở stage có nhãn).

Gọi \(f_1, f_2\) là `features` của hai view (đã `L2`-normalize theo kênh). Module `TC` trả temporal-contrasting loss và vector \(z\) (hoặc \(h\) ở SupCon) từ `projection_head`.

**Stage 1 — self-supervised (TS-TCC và CA-TCC):**

\[
\mathcal{L}_{\mathrm{ssl}}
= \mathcal{L}_{\mathrm{tc}}(f_1, f_2) + \mathcal{L}_{\mathrm{tc}}(f_2, f_1)
+ 0.7\,\mathcal{L}_{\mathrm{NT\text{-}Xent}}(z_1, z_2)
\]

\(\mathcal{L}_{\mathrm{tc}}\) là NCE trên \(K = 6\) bước tương lai (`TC.timesteps`). \(\mathcal{L}_{\mathrm{NT\text{-}Xent}}\) dùng cosine similarity, nhiệt độ \(\tau = 0.2\), chia cho \(2B\).

**Stage 4 CA-TCC — SupCon trên pseudo-label:**

\[
\mathcal{L}_{\mathrm{sup}}
= 0.01\bigl(\mathcal{L}_{\mathrm{tc}}(f_1, f_2) + \mathcal{L}_{\mathrm{tc}}(f_2, f_1)\bigr)
+ 0.1\,\mathcal{L}_{\mathrm{SupCon}}(h_1, h_2, \hat{y})
\]

**Stage có nhãn** (`ft_1p`, `train_linear`):

\[
\mathcal{L}_{\mathrm{ce}} = \mathrm{CrossEntropy}(\mathrm{logits}(x), y)
\]

Hai view và `TC` không vào loss này. `ReduceLROnPlateau` chỉ bước ở các stage có nhãn.

## Parameter updates

Gọi \(\theta_{\mathrm{ViT}}\) là backbone pretrained, \(\theta_{\mathrm{proj}}\) là `feature_proj`, \(\theta_{\mathrm{head}}\) là `logits`, \(\theta_{\mathrm{TC}}\) là module temporal contrasting.

Optimizer là Adam trên **đúng parameter còn `requires_grad`** (`filter` trong `_optimizers`). Cập nhật:

\[
\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L},
\qquad \theta \in \{\theta : \texttt{requires\_grad}=\mathrm{true}\}
\]

CNN không có \(\theta_{\mathrm{ViT}}\). Toàn bộ conv học từ stage 1. Đến linear probe, `_freeze_encoder` chỉ giữ gradient cho tên chứa `logits`.

DINOv3 **mặc định** (`không` `--finetune-encoder`) đóng ViT ngay trong `build_encoder`: `requires_grad=False`, `eval()`, và `forward` bọc `torch.no_grad()`. `train()` bị ghi đè để `Trainer` gọi `model.train()` mỗi epoch không kéo ViT lại train mode (DropPath sẽ làm feature pretrained ngẫu nhiên). Do đó với mọi stage:

\[
\nabla_{\theta_{\mathrm{ViT}}}\mathcal{L} = 0
\]

| Stage | CNN cập nhật | DINOv3 cập nhật (mặc định) |
| --- | --- | --- |
| 1 self-supervised | conv + \(\theta_{\mathrm{TC}}\) | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{TC}}\) |
| 2 CA-TCC `ft_1p` (1% nhãn) | conv + head | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{head}}\) |
| 4 SupCon | conv + \(\theta_{\mathrm{TC}}\) | \(\theta_{\mathrm{proj}}\), \(\theta_{\mathrm{TC}}\) |
| 5 linear probe | chỉ head | chỉ \(\theta_{\mathrm{head}}\) (`feature_proj` đóng ở `_maybe_freeze_for_linear`) |

Learning rate chung cho encoder và `TC` (chưa tách param group): CNN và DINOv3 đóng backbone dùng \(\eta = 3\cdot 10^{-4}\). Bật `--finetune-encoder` thì \(\theta_{\mathrm{ViT}}\) mở từ stage 1 và \(\eta\) hạ còn \(10^{-5}\) (`_encoder_lr`). Stage 5 lúc đó không freeze thêm.

`--batch-size` để trống: CNN 128, DINOv3 16. Một forward ViT trên ảnh 224 nặng hơn ba conv trên cửa sổ 128.

## Augmentation (chỗ còn lại phải đổi)

View yếu giữ jitter rồi scale, giống CNN. View mạnh thì không.

CNN cắt chuỗi rồi xáo thứ tự (`permutation` + jitter \(\sigma = 0.8\)). DINOv3 cộng vector vị trí cố định cho từng patch; xáo thời gian khiến đoạn sau đứng vào vị trí đoạn trước, và contrastive loss sẽ ép hai thứ tự khác nhau phải giống nhau. View mạnh của DINOv3 vì vậy là jitter nhẹ (\(\sigma = 0.2\)) rồi che khoảng 40% đoạn thời gian (`mask_patch_len = 8`, `mask_patch_stride = 8`), giữ đúng chỗ. PatchTST đã làm vậy vì cùng lý do. Hệ số nằm riêng trong `AugmentConfig` (`dinov3_jitter_ratio`, `dinov3_mask_ratio`).

## Cách chạy

```bash
uv run python scripts/train.py --encoder dinov3 --method ca-tcc --dinov3-variant vits16 --img-size 224
```

Fine-tune cả ViT:

```bash
uv run python scripts/train.py --encoder dinov3 --finetune-encoder --dinov3-variant vits16
```

Ảnh nhỏ hơn khi thiếu bộ nhớ, miễn chia hết cho 16 và \(N > 6\). Cạnh 64 là mức nhỏ nhất với `timesteps = 6`:

```bash
uv run python scripts/train.py --encoder dinov3 --img-size 64 --batch-size 32
```

Thư mục artifact vẫn `{method}_{encoder}`, nên lần chạy DINOv3 không ghi đè CNN.
