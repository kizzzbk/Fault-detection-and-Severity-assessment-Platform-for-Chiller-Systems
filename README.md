# Thay encoder bằng DINOv3

DINOv3 được thêm như một encoder thứ ba, cạnh CNN (`base_Model`) và PatchTST. Chọn nó bằng `--encoder dinov3`. CNN và PatchTST vẫn chạy như cũ.

Mục đích là dùng một ViT đã học sẵn trên ảnh làm phần rút đặc trưng cho cửa sổ cảm biến chiller, trong khi vòng train TS-TCC / CA-TCC không phải viết lại. Vì vậy encoder mới vẫn trả đúng contract mà module temporal contrasting đang đọc: `forward(x)` cho `(logits, features)`, với `features` có shape `(B, final_out_channels, N)` và `N` lớn hơn `TC.timesteps` (mặc định 6).

DINOv3 bản thân chỉ là backbone. Nó nhận ảnh 3 kênh và trả token, không trả logit lớp lỗi. Lớp chiếu token và head phân loại nằm trong encoder mới để khít contract đó.

## Contract và ý nghĩa của `N`

Input vẫn là cửa sổ `(B, C, T)`. Với cấu hình mặc định thì `C = 19` cột cảm biến và `T = 128`.

`logits` có shape `(B, num_classes)`. Đây là đầu ra của head, dùng khi train có nhãn, khi đánh giá, và khi gán pseudo-label.

`features` có shape `(B, 128, N)`. `128` là `final_out_channels`, số kênh mà module `TC` đang chờ. `N` với DINOv3 không còn là 18 của CNN. `N` là số patch của ViT: `(img_size / patch_size) ** 2`. Ảnh mặc định 224 và patch 16 cho lưới 14×14, nên `N = 196`.

## `src/chiller_fault_detection/models/dinov3.py`

File này là encoder mới. Nó không thay file CNN. `base_Model` vẫn được gọi khi `--encoder cnn`.

`HF_MODEL_IDS` map tên ngắn (`vits16`, `vitb16`, …) sang repo Hugging Face `facebook/dinov3-*-pretrain-lvd1689m`. Mặc định là `vits16` (khoảng 21 triệu tham số, embedding 384) để lần train đầu không phải tải ViT lớn. `vitb16` có embedding 768.

`n_dino_tokens(img_size, patch_size)` tính `N` trước khi tải trọng số. `TrainConfig` gọi hàm này để điền `features_len`. Ảnh phải chia hết cho patch size. Với `timesteps = 6`, cạnh ảnh nhỏ nhất dùng được là 64, vì 64/16 = 4 và `N = 16`. Cạnh 32 chỉ ra 4 token và bị `TrainConfig` từ chối.

`load_dinov3_backbone(variant)` gọi `transformers.AutoModel.from_pretrained`. Lần đầu sẽ tải checkpoint. Repo của Meta trên Hugging Face đang gated: cần chấp nhận license trên trang model rồi `huggingface-cli login`. Nếu thiếu gói hoặc chưa đăng nhập, hàm báo lỗi chỉ rõ việc đó.

`Dinov3Encoder.forward` làm bốn việc nối tiếp. Đầu tiên `_windows_to_images` co từng cửa sổ về `[0, 1]`, coi 19 kênh là chiều cao và 128 bước là chiều rộng, phóng lên ảnh vuông, lặp thành 3 kênh, rồi chuẩn hóa ImageNet. 19 kênh cảm biến không được đưa thẳng vào ViT như ảnh RGB. Tiếp theo backbone đọc ảnh đó. Token CLS và 4 register token bị bỏ, chỉ patch token được giữ. `feature_proj` chiếu mỗi token từ chiều embedding của ViT (384 với `vits16`) về `final_out_channels` (128) bằng conv 1×1, BatchNorm và GELU, ra `features` dạng `(B, 128, N)`. Cuối cùng `self.logits`, một `Linear`, flatten map đó thành logit lớp. Tên module có chữ `logits` vì bước linear probe chỉ giữ gradient cho parameter mang tên này.

`freeze_backbone` tắt gradient của ViT và chuyển nó sang eval. `train()` được ghi đè vì `Trainer` gọi `model.train()` mỗi vòng. Nếu không ghi đè, ViT bị kéo lại train mode và DropPath làm feature pretrained chạy ngẫu nhiên dù trọng số đã đóng. Khi backbone đóng, `_hidden` còn bọc forward trong `torch.no_grad()` để không xây graph qua hàng chục triệu tham số.

## `trainer.py`

`Dinov3Config` giữ những thứ chỉ DINOv3 dùng: biến thể, cạnh ảnh, patch size, và độ dài đoạn bị che ở strong view. CNN và PatchTST không đọc dataclass này.

`TrainConfig.dinov3` mang config đó. `__post_init__` thêm nhánh `encoder == "dinov3"` và gán `features_len = n_dino_tokens(...)`. Không có nhánh này thì tên `dinov3` bị coi là encoder lạ ngay lúc tạo config. Kiểm tra `features_len > TC.timesteps` dùng chung, nên ảnh quá nhỏ vẫn bị chặn.

`build_encoder` trả `Dinov3Encoder`. Nếu không bật `--finetune-encoder`, hàm gọi `freeze_backbone()` ngay tại đây, trước stage 1. Mục đích là stage self-supervised không được ghi đè trọng số pretrained. Lớp `feature_proj` và `logits` vẫn học được. Nếu bật fine-tune, ViT mở gradient và learning rate bị hạ.

`_encoder_lr` thêm một nhánh. PatchTST vẫn tối đa `1e-4`. DINOv3 chỉ bị kéo xuống `1e-5` khi ViT thực sự được cập nhật. Backbone đóng thì projection, head và module `TC` giữ `3e-4`. Cùng một learning rate đang được dùng cho cả encoder lẫn `TC`, nên hạ rate của ViT cũng hạ rate của `TC`. Chưa tách param group.

`_freeze_encoder` không sửa. Nó vẫn tắt mọi parameter trừ tên có chữ `logits`.

`_maybe_freeze_for_linear` vẫn là cổng của stage cuối. PatchTST, hoặc bất kỳ encoder nào bật `--finetune-encoder`, train toàn bộ. CNN và DINOv3 không bật cờ thì rơi vào `_freeze_encoder`. Với DINOv3, lần freeze này xảy ra sau khi projection đã được các stage contrastive học xong, nên stage cuối chỉ cập nhật head trên nhãn thật. ViT thì đã đóng từ `build_encoder`.

`_experiment_header` ghi thêm biến thể, cạnh ảnh, số token, learning rate và việc backbone có bị đóng hay không, để report phân biệt các lần chạy DINOv3.

Các hàm `run` và `run_ca_tcc` không đổi thứ tự stage. Chúng đã gọi `build_encoder` và `_maybe_freeze_for_linear`, nên lịch freeze ở trên áp vào cả TS-TCC và CA-TCC.

Lịch đó, khi không fine-tune, diễn ra như sau. Từ stage 1, ViT đứng yên. Stage self-supervised và SupCon cập nhật `feature_proj` cùng module `TC`, vì loss lấy `features`. Stage có nhãn (fine-tune trên một phần nhãn, rồi linear probe) cập nhật `logits`. Đến stage linear probe, `feature_proj` cũng bị đóng và chỉ còn head.

Khi `--finetune-encoder`, ViT học từ stage 1 với learning rate `1e-5`, và stage cuối không đóng gì thêm.

## `augmentations.py` và `AugmentConfig`

`DataTransform` thêm nhánh `encoder == "dinov3"`. Bản yếu vẫn là jitter rồi scale, giống mọi encoder. Bản mạnh là jitter nhẹ rồi che một phần các đoạn thời gian (`mask_patch_len = 8`, `mask_patch_stride = 8`, che khoảng 40%).

Nhánh này tồn tại vì bản mạnh của CNN cắt chuỗi rồi xáo thứ tự. CNN nhìn cục bộ nên xáo vẫn còn dạng sóng. DINOv3 cộng vector vị trí cố định cho từng patch. Xáo thời gian khiến nội dung của đoạn sau đứng vào vị trí của đoạn trước, và contrastive loss sẽ ép hai chuỗi khác nhau phải giống nhau. Che đoạn và giữ đúng chỗ thì bản mạnh vẫn là cùng một cửa sổ. PatchTST đã làm vậy vì cùng lý do. Hệ số jitter và mask của DINOv3 nằm riêng trong `AugmentConfig` (`dinov3_jitter_ratio`, `dinov3_mask_ratio`) để không dùng sigma 0.8 của CNN.

## `scripts/train.py`

`--encoder` nhận thêm `dinov3`. `--dinov3-variant` chọn `vits16`, `vits16plus`, `vitb16` hoặc `vitl16`. `--img-size` là cạnh ảnh vuông, mặc định 224, phải chia hết cho 16. Hai cờ này chỉ có tác dụng khi encoder là DINOv3. Chúng được gói vào `Dinov3Config` rồi đưa vào `TrainConfig`.

`--batch-size` để trống thì CNN và PatchTST vẫn dùng 128, còn DINOv3 dùng 16. Một forward ViT trên ảnh 224 nặng hơn CNN cửa sổ 128, nên batch 128 dễ hết bộ nhớ. Truyền `--batch-size` thì số đó được giữ nguyên.

`--finetune-encoder` với DINOv3 không chỉ mở head ở stage cuối. Nó mở cả ViT từ stage 1.

Tên thư mục artifact vẫn là `{method}_{encoder}`, nên một lần chạy DINOv3 không ghi đè artifact của CNN.

## `pyproject.toml`

Thêm dependency `transformers>=4.56`. `AutoModel` của gói này là cách tải backbone. PyTorch đã có sẵn, không đủ để đọc checkpoint Hugging Face.

## Chỗ chỉ đổi lời mô tả

`model.py` vẫn là CNN. Dòng đầu file ghi thêm rằng DINOv3 là encoder thứ ba và cùng contract.

`fixmatch.py` không đổi thuật toán. Comment stage 5 được sửa cho khớp: CNN và DINOv3 train head, PatchTST và `--finetune-encoder` train cả mạng. FixMatch gọi `build_encoder` và `_maybe_freeze_for_linear`, nên nó nhận lịch freeze mới mà không cần sửa vòng lặp.

## File không đổi

`TC.py`, `loss.py` và `attention.py` không đổi vì chúng chỉ thấy `features` dạng `(B, final_out_channels, N)`. Chiếu token về 128 kênh là để giữ được điều đó.

`data/preprocess.py` và `data/load.py` không đổi. Cửa sổ vẫn là `(B, C, T)`. Việc vẽ cửa sổ thành ảnh nằm trong `forward` của encoder.

## Cách chạy

```bash
uv run python scripts/train.py --encoder dinov3 --method ca-tcc --dinov3-variant vits16 --img-size 224
```

Fine-tune cả ViT:

```bash
uv run python scripts/train.py --encoder dinov3 --finetune-encoder --dinov3-variant vits16
```

Ảnh nhỏ hơn khi thiếu bộ nhớ, miễn là chia hết cho 16 và `N > 6`. Cạnh 64 là mức nhỏ nhất với `timesteps = 6`:

```bash
uv run python scripts/train.py --encoder dinov3 --img-size 64 --batch-size 32
```
