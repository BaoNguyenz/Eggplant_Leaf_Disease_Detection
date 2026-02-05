# ViT Feature Extraction System

Hệ thống trích xuất đặc trưng sử dụng **Vision Transformer (ViT-B/16)** cho bài toán phân loại bệnh lá cà tím.

## 📋 Tổng quan

### Kiến trúc Model
- **Model:** `vit_b_16` (Vision Transformer Base, patch size 16)
- **Pretrained Weights:** `IMAGENET1K_V1` (ImageNet-1K)
- **Feature Vector:** 768 chiều (từ [CLS] token)
- **Input Size:** 224 × 224 pixels

### Đặc điểm kỹ thuật
- ✅ **Trích xuất từ [CLS] token** (chuẩn ViT gốc)
- ✅ **Batch processing** để tối ưu tốc độ
- ✅ **GPU support** (CUDA tự động phát hiện)
- ✅ **UTF-8 encoding** cho Windows
- ✅ **Modular architecture** (5 files)

## 🗂️ Cấu trúc thư mục

```
vit_feature_extraction/
├── config.py          # Cấu hình (paths, hyperparameters)
├── dataset.py         # Dataset loader & transforms
├── model.py           # ViT model wrapper
├── utils.py           # Helper functions
├── extract.py         # Main script (entry point)
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## 🔧 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Kiểm tra CUDA (optional)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 🚀 Sử dụng

### Cách 1: Sử dụng default paths (từ config.py)

```bash
python extract.py
```

### Cách 2: Custom paths

```bash
python extract.py --data_dir "path/to/images" --output_dir "path/to/output"
```

### Cách 3: Sử dụng Custom Trained Weights (Recommended)

```bash
# Sử dụng custom weights từ config.py (mặc định)
python extract.py

# Hoặc chỉ định path trực tiếp
python extract.py --custom_weights "path/to/best_model.pth"
```

**Lưu ý:** Trong `config.py`, đặt:
```python
USE_CUSTOM_WEIGHTS = True  # Sử dụng custom weights
CUSTOM_WEIGHTS_PATH = Path(r"E:\...\vit_output\best_model.pth")
```

### Cách 4: Quay lại ImageNet Pretrained Weights

```bash
# Bỏ qua custom weights, dùng ImageNet
python extract.py --use_imagenet
```

### Cách 5: Custom batch size

```bash
python extract.py --batch_size 64 --output_csv "custom_features.csv"
```

### Tất cả tham số

```bash
python extract.py \
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" \
  --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction" \
  --batch_size 32 \
  --output_csv "vit_features.csv" \
  --custom_weights "E:\...\vit_output\best_model.pth"
```

## 📊 Output Format

### CSV Structure

| Column | Description |
|--------|-------------|
| `image_path` | Absolute path to image file |
| `class_name` | Class label (folder name) |
| `feature_1` ... `feature_768` | Feature vector (768 dimensions) |

### Example

```csv
image_path,class_name,feature_1,feature_2,...,feature_768
E:\...\Healthy\img001.jpg,Healthy,0.234,-0.567,...,1.234
E:\...\Disease_A\img002.jpg,Disease_A,-0.123,0.456,...,-0.789
```

## 🧠 Kiến trúc ViT

### Tại sao sử dụng [CLS] token?

```python
# Trong model.py - forward() method:

# Encoder output shape: [batch_size, 197, 768]
# - 197 tokens = 1 [CLS] + 196 patches (14×14 grid)
# - 768 = hidden dimension

features = x[:, 0]  # Extract [CLS] token
# Output shape: [batch_size, 768]
```

**Lý do:**
1. **Chuẩn ViT gốc** (Dosovitskiy et al., ICLR 2021)
2. **Pre-trained weights** được tối ưu cho representation này
3. **[CLS] token tổng hợp thông tin** từ tất cả patches qua self-attention
4. **Đầu vào classification head** trong quá trình pre-train

**Alternatives (KHÔNG dùng):**
- `x[:, 1:].mean(dim=1)` - Average pooling (bỏ qua pre-training)
- `x.mean(dim=1)` - Average all tokens (non-standard)

## 📈 Performance

### Tốc độ xử lý (ước tính)
- **GPU (CUDA):** ~100-200 images/second
- **CPU:** ~10-20 images/second

### Memory usage
- **Batch size 32:** ~4-6 GB GPU memory
- **Batch size 64:** ~8-12 GB GPU memory

## 🔍 Technical Details

### Image Preprocessing

```python
transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to ViT input size
    transforms.ToTensor(),                   # Convert to [0, 1]
    transforms.Normalize(                    # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Feature Extraction Pipeline

1. **Input:** RGB image (any size)
2. **Resize:** 224 × 224
3. **Normalize:** ImageNet mean/std
4. **Patch embedding:** 16×16 patches → 196 tokens
5. **Add [CLS] token:** Position 0
6. **Positional encoding:** Learned embeddings
7. **Transformer encoder:** 12 blocks
8. **Extract [CLS]:** 768-dimensional vector

## 📝 Code Style

- ✅ **PEP 8** compliant
- ✅ **Type Hints** (Python 3.10+)
- ✅ **Docstrings** (Google style)
- ✅ **Error handling** (try-except)
- ✅ **Clean Code** principles

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
**Solution:** Giảm batch size
```bash
python extract.py --batch_size 16
```

### Error: "No images found"
**Solution:** Kiểm tra data_dir path và directory structure
```
data_dir/
├── Class1/
│   └── *.jpg
├── Class2/
│   └── *.jpg
```

### Error: "Module not found"
**Solution:** Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 📚 References

1. **ViT Paper:** Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. **PyTorch Documentation:** https://pytorch.org/vision/stable/models/vision_transformer.html
3. **Pretrained Weights:** https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html

## 📧 Contact

Developed by: AI Engineer  
Project: Eggplant Leaf Disease Detection  
Date: 2026-01-28
