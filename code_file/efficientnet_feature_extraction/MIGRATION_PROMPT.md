# Prompt: Migrate EfficientNet Feature Extraction from B7 to B0

## Context

Tôi có một hệ thống feature extraction modular sử dụng EfficientNet-B7 pretrained để trích xuất features từ ảnh lá cà tím. Hệ thống gồm 5 file Python chính:
- `config.py` - Quản lý cấu hình
- `model.py` - Định nghĩa model architecture
- `dataset.py` - Custom dataset và transforms
- `utils.py` - Utility functions
- `extract.py` - Main entry point

## Task Request

Hãy chuyển đổi toàn bộ hệ thống từ **EfficientNet-B7** sang **EfficientNet-B0** với các yêu cầu sau:

### 1. Custom Weights
- Sử dụng custom trained EfficientNet-B0 weights từ checkpoint:
  ```
  E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction\effcientnetb0_output\best_model.pth
  ```
- Đặt đường dẫn này làm default trong `Config.CUSTOM_WEIGHTS_PATH`

### 2. Architecture Changes

**config.py:**
- `IMG_SIZE`: Thay đổi từ `(600, 600)` → `(224, 224)`
- `FEATURE_DIM`: Thay đổi từ `2560` → `1280`
- `BATCH_SIZE`: Thay đổi từ `4` → `8` (vì B0 nhẹ hơn)
- Cập nhật tất cả docstrings từ B7 → B0

**model.py:**
- Import: `from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights`
- Class name: `EfficientNetB7FeatureExtractor` → `EfficientNetB0FeatureExtractor`
- Weights: `EfficientNet_B7_Weights.IMAGENET1K_V1` → `EfficientNet_B0_Weights.IMAGENET1K_V1`
- Feature dimension: `self.feature_dim = 2560` → `self.feature_dim = 1280`
- Model loading: `efficientnet_b7(weights=weights)` → `efficientnet_b0(weights=weights)`
- Cập nhật tất cả comments về:
  - Input shape: `[batch_size, 3, 600, 600]` → `[batch_size, 3, 224, 224]`
  - Feature shape: `[batch_size, 2560, H', W']` → `[batch_size, 1280, H', W']`
  - Output shape: `[batch_size, 2560]` → `[batch_size, 1280]`
- Verify output shape assertion: `assert output.shape == (1, 2560)` → `assert output.shape == (1, 1280)`
- Update function signatures: `EfficientNetB7FeatureExtractor` → `EfficientNetB0FeatureExtractor`

**dataset.py:**
- Function name: `get_efficientnet_b7_transforms()` → `get_efficientnet_b0_transforms()`
- Default img_size: `(600, 600)` → `(224, 224)`
- Cập nhật docstrings và comments từ B7 → B0

**extract.py:**
- Import: `get_efficientnet_b7_transforms` → `get_efficientnet_b0_transforms`
- Argparse description: "EfficientNet-B7" → "EfficientNet-B0"
- Default output filename: `efficientnet_b7_features.csv` → `efficientnet_b0_features.csv`
- Cập nhật tất cả print statements và comments

**example_usage.py:**
- Default CSV path: `efficientnet_b7_features.csv` → `efficientnet_b0_features.csv`

### 3. Verification Requirements

Sau khi chỉnh sửa, cần verify:
1. Model load được custom weights thành công
2. Output shape đúng là `[batch, 1280]`
3. Chạy được feature extraction trên toàn bộ dataset

**Test command:**
```powershell
conda activate torch
python -c "from model import load_feature_extractor, verify_output_shape; from config import Config; model = load_feature_extractor(Config.DEVICE, pretrained=False, custom_weights_path=Config.CUSTOM_WEIGHTS_PATH); verify_output_shape(model, Config.IMG_SIZE)"
```

**Expected output:**
- Checkpoint loads successfully
- Output shape: `torch.Size([1, 1280])` ✓
- Verification PASSED

### 4. Output Format

File CSV output sẽ có format:
- Column 1: `image_path`
- Column 2: `class_name`
- Columns 3-1282: `feature_1` đến `feature_1280` (giảm từ 2560 xuống 1280)

### 5. Important Notes

> **Breaking Changes:**
> - Feature dimension giảm 50% (2560 → 1280)
> - Input resolution giảm đáng kể (600×600 → 224×224)
> - Batch size có thể tăng lên (4 → 8 hoặc cao hơn)

> **Advantages:**
> - Faster inference (~5-10× faster than B7)
> - Lower memory usage (~3× less GPU memory)
> - Domain-specific features (custom trained on eggplant dataset)

### 6. Structure Location

```
E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction\
├── config.py           # Update parameters
├── model.py            # Update architecture
├── dataset.py          # Update transforms
├── extract.py          # Update main script
├── utils.py            # No changes needed
├── example_usage.py    # Update CSV path
└── effcientnetb0_output\
    └── best_model.pth  # Custom weights checkpoint
```

## Expected Deliverables

1. ✅ Tất cả 5 files được cập nhật với B0 parameters
2. ✅ Model verification test pass
3. ✅ Feature extraction chạy thành công
4. ✅ Output CSV với 1280 feature dimensions

## Success Criteria

Feature extraction hoàn tất với output:
```
Total images   : 4089
Feature shape  : torch.Size([4089, 1280])
Output CSV     : efficientnet_b0_features.csv
File size      : ~114 MB
```
