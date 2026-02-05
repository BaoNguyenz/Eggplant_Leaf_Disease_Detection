# EfficientNet-B7 Feature Extraction System

## 📋 Tổng Quan

Hệ thống trích xuất đặc trưng **modular** sử dụng **EfficientNet-B7** pretrained cho dataset **Eggplant Leaf Disease Detection**. Được tối ưu hoá cho **Windows 10/11** với xử lý đường dẫn sâu và encoding UTF-8.

### ✨ Đặc Điểm Chính

- **Kiến trúc Modular**: 5 files Python độc lập, dễ bảo trì
- **Windows Optimized**: `pathlib.Path` cho tất cả thao tác đường dẫn
- **EfficientNet-B7**: Pretrained ImageNet, feature vector **2560 chiều**
- **Global Average Pooling**: Chuẩn hóa feature map từ convolution
- **CLI Support**: Argparse với tham số linh hoạt

---

## 📂 Cấu Trúc Thư Mục

```
E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\
│
├── Eggplant Dataset\
│   └── Classified Images\              # Dataset input
│       ├── Class_A\
│       │   ├── image1.jpg
│       │   └── image2.jpg
│       ├── Class_B\
│       └── ...
│
└── code_file\
    └── efficientnet_feature_extraction\    # Thư mục project
        ├── config.py                        # Cấu hình đường dẫn & tham số
        ├── utils.py                         # Hàm phụ trợ (CSV, path, validation)
        ├── dataset.py                       # Custom Dataset & Transforms
        ├── model.py                         # EfficientNet-B7 Feature Extractor
        ├── extract.py                       # Entry point (CLI)
        ├── requirements.txt                 # Dependencies
        ├── README.md                        # Tài liệu này
        └── efficientnet_b7_features.csv     # Output (sau khi chạy)
```

---

## 🛠️ Kiến Trúc Hệ Thống

```mermaid
graph TD
    A[extract.py<br/>Entry Point] --> B[config.py<br/>Configuration]
    A --> C[dataset.py<br/>Dataset & Transforms]
    A --> D[model.py<br/>EfficientNet-B7]
    A --> E[utils.py<br/>Utilities]
    
    C --> F[DataLoader]
    D --> G[Feature Extractor<br/>2560-dim]
    F --> G
    G --> H[Features Tensor<br/>[N, 2560]]
    E --> I[CSV Writer<br/>UTF-8]
    H --> I
    I --> J[efficientnet_b7_features.csv]
    
    style A fill:#4CAF50,color:#fff
    style G fill:#2196F3,color:#fff
    style J fill:#FF9800,color:#fff
```

---

## 🔧 Chi Tiết Module

### 1. **config.py** - Configuration Manager
Quản lý tập trung tất cả đường dẫn và tham số:
- `DATA_DIR`: Đường dẫn dataset (Classified Images)
- `OUTPUT_DIR`: Thư mục lưu CSV
- `BATCH_SIZE`: Mặc định **4** (tối ưu cho GPU limited)
- `IMG_SIZE`: **(600, 600)** - Resolution chuẩn cho EfficientNet-B7
- `FEATURE_DIM`: **2560** chiều
- `DEVICE`: Auto-detect CUDA/CPU
- `IMAGENET_MEAN/STD`: Normalization ImageNet

**Methods:**
- `update_paths()`: Cập nhật paths từ CLI
- `validate_paths()`: Kiểm tra tính hợp lệ
- `get_info()`: In thông tin cấu hình

---

### 2. **utils.py** - Utility Functions
Các hàm phụ trợ cho pipeline:

| Function | Mô Tả |
|----------|-------|
| `ensure_dir(directory)` | Tạo thư mục nếu chưa tồn tại |
| `get_image_paths(data_dir)` | Lấy list tất cả đường dẫn ảnh |
| `extract_class_from_path(img_path)` | Trích xuất class name từ parent folder |
| `save_features_to_csv(...)` | Lưu features vào CSV (UTF-8 encoding) |
| `validate_features(features)` | Kiểm tra NaN, Inf, dimension |
| `print_device_info()` | In thông tin GPU/CPU |

**Windows Optimization:**
- Sử dụng `pathlib.Path` cho cross-platform
- UTF-8 encoding khi ghi CSV
- Convert Windows path (`\`) thành forward slash (`/`) trong CSV

---

### 3. **dataset.py** - Custom Dataset & Transforms

#### **EggplantLeafDataset**
Custom PyTorch Dataset hỗ trợ:
- Load ảnh từ cấu trúc thư mục phân cấp (class/image.jpg)
- Convert PIL Image → RGB
- Áp dụng transforms pipeline

#### **get_efficientnet_b7_transforms()**
Transform pipeline chuẩn cho EfficientNet-B7:

```python
1. Resize((600, 600))         # Resolution tối ưu
2. ToTensor()                 # Convert to [0, 1] float tensor
3. Normalize(ImageNet mean/std)  # Chuẩn hóa với ImageNet statistics
```

---

### 4. **model.py** - EfficientNet-B7 Feature Extractor

#### **EfficientNetB7FeatureExtractor**
Kiến trúc trích xuất đặc trưng:

```
Input: [batch_size, 3, 600, 600]
  ↓
efficientnet.features (Convolution Blocks)
  ↓
[batch_size, 2560, H', W']
  ↓
Global Average Pooling (AdaptiveAvgPool2d)
  ↓
[batch_size, 2560, 1, 1]
  ↓
Flatten
  ↓
Output: [batch_size, 2560]
```

**Đặc điểm kỹ thuật:**
- **Pretrained Weights**: `EfficientNet_B7_Weights.IMAGENET1K_V1`
- **Feature Extraction Only**: Loại bỏ hoàn toàn `classifier` layer
- **Global Average Pooling**: Giảm spatial dimensions → vector 1D
- **Total Parameters**: ~66M (chỉ tính phần features)

**Methods:**
- `forward(x)`: Trích xuất features
- `get_feature_dim()`: Trả về 2560
- `verify_output_shape()`: Test với dummy input

---

### 5. **extract.py** - Main Entry Point

Script chạy chính với **CLI arguments**:

```bash
python extract.py \
  --data_dir "path/to/Classified Images" \
  --output_dir "path/to/output" \
  --batch_size 4 \
  --num_workers 4 \
  --output_filename "efficientnet_b7_features.csv"
```

**Pipeline Execution:**
1. Parse CLI arguments
2. Validate paths và config
3. Load dataset (scan tất cả class folders)
4. Create DataLoader với batch size tối ưu
5. Load EfficientNet-B7 pretrained model
6. Extract features (với progress bar)
7. Validate output (check NaN, Inf, dimensions)
8. Save to CSV (UTF-8 encoding)

---

## 📦 Cài Đặt

### 1. Dependencies

Tạo file `requirements.txt`:

```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
tqdm>=4.60.0
```

### 2. Cài đặt packages

```powershell
pip install -r requirements.txt
```

> **Lưu ý**: Nếu có GPU NVIDIA, cài đặt PyTorch với CUDA support:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. **Chuẩn Bị Dataset**

Đảm bảo dataset có cấu trúc:

```
Classified Images/
├── Bacterial_Wilt/
│   ├── img1.jpg
│   └── img2.jpg
├── Healthy/
│   ├── img3.jpg
│   └── img4.jpg
└── Phomopsis_Blight/
    └── img5.jpg
```

---

### 2. **Chạy Feature Extraction**

#### **Option A: Sử dụng Default Paths**

```powershell
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction"
python extract.py
```

#### **Option B: Custom Paths**

```powershell
python extract.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction" `
  --batch_size 2 `
  --num_workers 2 `
  --output_filename "my_features.csv"
```

> **Tip cho GPU limited**: Giảm `--batch_size` xuống **2** hoặc **1** nếu gặp lỗi `CUDA Out of Memory`.

---

### 3. **Ví Dụ Lệnh Chạy Đầy Đủ**

```powershell
# Mở PowerShell tại thư mục project
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction"

# Chạy extraction với batch size thấp (an toàn cho GPU 6GB)
python extract.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction" `
  --batch_size 2 `
  --num_workers 4 `
  --output_filename "efficientnet_b7_features.csv"
```

---

## 📊 Output Format

### CSV Structure

File **efficientnet_b7_features.csv** có format:

| image_path | class_name | feature_1 | feature_2 | ... | feature_2560 |
|------------|------------|-----------|-----------|-----|--------------|
| Healthy/img1.jpg | Healthy | 0.234 | -0.145 | ... | 0.678 |
| Bacterial_Wilt/img2.jpg | Bacterial_Wilt | -0.023 | 0.567 | ... | -0.234 |

**Columns:**
- `image_path`: Relative path từ Classified Images (forward slash)
- `class_name`: Tên class (parent folder name)
- `feature_1` → `feature_2560`: 2560 chiều feature vector

**Encoding**: UTF-8 (hỗ trợ tiếng Việt và ký tự đặc biệt)

---

## ⚙️ Tham Số Tối Ưu

| Parameter | Recommended Value | Giải Thích |
|-----------|-------------------|------------|
| `--batch_size` | **2-4** | Thấp do EfficientNet-B7 nặng (66M params) |
| `--num_workers` | **4** | Tùy số CPU cores (2-8) |
| `IMG_SIZE` | **(600, 600)** | Resolution tối ưu cho EfficientNet-B7 |
| `FEATURE_DIM` | **2560** | Fixed (architecture của B7) |

**GPU Memory Usage Estimate:**
- Batch size **4**: ~8-10 GB VRAM
- Batch size **2**: ~4-5 GB VRAM
- Batch size **1**: ~2-3 GB VRAM

---

## 🐛 Troubleshooting

### ❌ **Lỗi: CUDA Out of Memory**

**Giải pháp:**
```powershell
python extract.py --batch_size 1
```

---

### ❌ **Lỗi: FileNotFoundError**

**Nguyên nhân:** Đường dẫn `data_dir` không tồn tại.

**Giải pháp:**
1. Kiểm tra đường dẫn với PowerShell:
   ```powershell
   Test-Path "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images"
   ```
2. Sửa lại tham số `--data_dir`

---

### ❌ **Lỗi: UnicodeDecodeError khi đọc CSV**

**Giải pháp:** File đã được lưu với UTF-8. Khi đọc, chỉ định encoding:
```python
import pandas as pd
df = pd.read_csv("efficientnet_b7_features.csv", encoding='utf-8')
```

---

## 📈 Sử Dụng Features Đã Trích Xuất

### Ví dụ: Load và Visualize với Pandas

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("efficientnet_b7_features.csv", encoding='utf-8')

# Hiển thị thông tin
print(f"Total images: {len(df)}")
print(f"Classes: {df['class_name'].unique()}")

# Trích xuất feature matrix
features = df.iloc[:, 2:].values  # Bỏ 2 cột đầu (path, class)
print(f"Feature matrix shape: {features.shape}")  # (N, 2560)

# Trích xuất labels
labels = df['class_name'].values
```

### Ví dụ: Train Classifier (SVM)

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM
svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
svm_clf.fit(X_train, y_train)

# Evaluate
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

---

## 📚 Technical References

- **EfficientNet Paper**: [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- **PyTorch EfficientNet**: [torchvision.models.efficientnet_b7](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html)
- **ImageNet Weights**: `EfficientNet_B7_Weights.IMAGENET1K_V1`

---

## 📝 License & Credits

**Developed for:** Eggplant Leaf Disease Detection Project  
**Architecture:** Modular Python (PEP 8)  
**Platform:** Windows 10/11 with CUDA Support  
**Model:** EfficientNet-B7 (Google Research)

---

## 🔄 Version History

- **v1.0** (2026-01-26): Initial release
  - Modular architecture (5 files)
  - EfficientNet-B7 with 2560-dim features
  - Windows path optimization
  - UTF-8 CSV encoding
  - CLI argparse support
