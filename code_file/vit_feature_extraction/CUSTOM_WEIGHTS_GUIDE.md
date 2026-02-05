# Custom Weights Support - Quick Guide

## ✅ Hệ thống đã được cập nhật

Bây giờ bạn có thể sử dụng **custom trained ViT weights** từ `best_model.pth` để trích xuất features!

---

## 🚀 Cách sử dụng

### Option 1: Sử dụng config (Đơn giản nhất - Recommended)

Trong `config.py`, đặt:
```python
USE_CUSTOM_WEIGHTS = True
CUSTOM_WEIGHTS_PATH = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction\vit_output\best_model.pth")
```

Sau đó chạy:
```bash
python extract.py
```

### Option 2: Chỉ định trực tiếp qua command line

```bash
python extract.py --custom_weights "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction\vit_output\best_model.pth"
```

### Option 3: Quay lại ImageNet pretrained weights

```bash
python extract.py --use_imagenet
```

Hoặc trong `config.py`:
```python
USE_CUSTOM_WEIGHTS = False
```

---

## 📊 Thông tin kỹ thuật

### Checkpoint structure
Checkpoint file được lưu với format:
```python
{
    'epoch': 57,
    'model_state_dict': {...},  # Model weights
    'optimizer_state_dict': {...},
    'metrics': {
        'accuracy': 0.5232,
        'precision': 0.5562,
        'recall': 0.5232,
        'f1': 0.5314
    }
}
```

### Xử lý tự động
Code tự động xử lý:
- ✅ **DataParallel prefix:** Loại bỏ `module.` prefix nếu có
- ✅ **Classification head:** Bỏ qua `heads.head.*` weights (task-specific)
- ✅ **Encoder weights:** Chỉ load encoder weights cho feature extraction
- ✅ **Metrics display:** Hiển thị metrics từ checkpoint

### Output console
```
[INFO] Loading custom ViT weights from: E:\...\vit_output\best_model.pth
[INFO] Loaded checkpoint from epoch 57
[INFO] Checkpoint metrics: {'accuracy': 0.5232, 'precision': 0.5562, 'recall': 0.5232, 'f1': 0.5314}
[INFO] Loading 150 weights (excluding classification head)
[SUCCESS] Custom weights loaded successfully!
```

---

## 🧪 Đã test thành công

```
======================================================================
Testing Custom Weights Loading
======================================================================

[TEST 1] Loading model with custom weights...
✓ Model loaded successfully!
✓ Feature dimension: 768

[TEST 2] Testing forward pass...
✓ Input shape: torch.Size([2, 3, 224, 224])
✓ Output shape: torch.Size([2, 768])
✓ Expected shape: torch.Size([2, 768])

✅ ALL TESTS PASSED!
```

---

## 📁 Files đã cập nhật

1. **config.py** - Thêm `USE_CUSTOM_WEIGHTS` và `CUSTOM_WEIGHTS_PATH`
2. **model.py** - Support loading custom weights với xử lý DataParallel
3. **extract.py** - Command-line arguments cho custom weights
4. **README.md** - Documentation đầy đủ

---

## 💡 Lưu ý quan trọng

1. **Custom weights được ưu tiên** nếu `USE_CUSTOM_WEIGHTS = True` trong config
2. **Command-line override config:** `--custom_weights` sẽ override config
3. **Force ImageNet:** Dùng `--use_imagenet` để bỏ qua custom weights
4. **Classification head bị bỏ qua:** Vì chúng ta chỉ cần encoder weights

---

## 🎯 Sử dụng ngay

```bash
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction"
python extract.py
```

Hệ thống sẽ tự động:
- ✅ Load custom weights từ `vit_output/best_model.pth`
- ✅ Hiển thị metrics từ checkpoint (epoch 57, F1=0.5314)
- ✅ Trích xuất features 768 chiều từ [CLS] token
- ✅ Lưu vào CSV với UTF-8 encoding

**Hoàn thành!** 🎉
