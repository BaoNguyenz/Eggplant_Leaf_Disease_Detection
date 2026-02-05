# ViT Modular Training Pipeline - Cấu trúc Dự án

## 📂 Cấu Trúc Thư Mục

```
E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\
│
├── Eggplant Dataset\
│   └── Classified Images\            ← Dataset gốc (6 lớp bệnh)
│       ├── Healthy Leaf\
│       ├── Insect_Pest_Disease\
│       ├── Leaf_Spot_Disease\
│       ├── Mosaic_Virus_Disease\
│       ├── White_Mold_Disease\
│       └── Wilt_Disease\
│
└── code_file\
    └── vit_pretrained\               ← Hệ thống huấn luyện modular
        ├── config.py                 (6,167 bytes) - Quản lý tham số CLI
        ├── dataset.py                (9,122 bytes) - Data loading & augmentation
        ├── model.py                  (2,983 bytes) - ViT-B/16 model
        ├── loss.py                   (5,449 bytes) - CrossEntropy & Focal Loss
        ├── utils.py                 (11,268 bytes) - Metrics & visualization
        ├── train.py                 (11,467 bytes) - Main training script
        ├── README.md                 (7,221 bytes) - Tài liệu đầy đủ
        └── example_commands.py       (4,392 bytes) - Các lệnh mẫu

        [Sau khi train sẽ tạo thư mục outputs/]
        └── outputs\
            ├── best_model.pth        ← Model tốt nhất (theo val F1)
            ├── training_history.json ← Lịch sử metrics
            ├── loss_curves.png       ← Đồ thị Loss
            ├── f1_curve.png          ← Đồ thị F1-Score
            ├── confusion_matrix.png  ← Ma trận nhầm lẫn
            └── training.log          ← Log chi tiết
```

## 📋 Chi Tiết Từng File

### 1. **config.py** (6.2 KB)
- **Chức năng**: Quản lý tham số từ CLI
- **Tính năng chính**:
  - 20+ tham số điều khiển toàn bộ quá trình training
  - Auto-detect CUDA/CPU
  - Validation đầu vào (kiểm tra train_split + val_split < 1.0)
  - `print_config()` để hiển thị cấu hình đẹp
- **Sử dụng**: `python config.py --help`

### 2. **dataset.py** (9.1 KB)
- **Chức năng**: Xử lý dữ liệu và augmentation
- **Tính năng chính**:
  - **Auto class weight calculation**: Tự động tính trọng số lớp từ train set
  - **Strong augmentation**: RandAugment, RandomRotation, ColorJitter, v.v.
  - **Stratified splitting**: Chia dữ liệu 80/10/10 (train/val/test)
  - **WeightedRandomSampler**: Oversample lớp thiểu số
- **Output**: Trả về 3 DataLoader + class_weights + class_names

### 3. **model.py** (3.0 KB)
- **Chức năng**: Tạo mô hình ViT
- **Tính năng chính**:
  - ViT-B/16 với ImageNet pretrained weights
  - Thay thế classification head (768 → 6 classes)
  - Optional freeze backbone (chỉ train head)
  - Đếm parameters (total, trainable, frozen)
- **Cấu trúc**: ~86M params (full trainable) hoặc ~4.6K (frozen)

### 4. **loss.py** (5.4 KB)
- **Chức năng**: Loss functions cho imbalanced data
- **Tính năng chính**:
  - **CrossEntropyLoss** với class weights
  - **Focal Loss** (custom implementation) với alpha và gamma
  - Loss factory: `get_loss_fn()`
- **Khi nào dùng**:
  - CrossEntropy: Imbalance vừa phải
  - Focal Loss: Imbalance nghiêm trọng

### 5. **utils.py** (11.3 KB)
- **Chức năng**: Utilities cho training
- **Tính năng chính**:
  - **EarlyStopping class**: Theo dõi val F1, tự động dừng
  - **Metrics**: Accuracy, Precision, Recall, F1-Score (weighted)
  - **Plotting**: Loss curves, F1 curve, Confusion matrix
  - **Logger setup**: File + console logging
  - **Checkpoint**: Save/load model
- **Visualizations**: 3 loại plot (matplotlib + seaborn)

### 6. **train.py** (11.5 KB)
- **Chức năng**: Orchestrate toàn bộ quá trình training
- **Pipeline**:
  1. Parse arguments
  2. Setup logging
  3. Load data (với class weights)
  4. Build model
  5. Setup optimizer + scheduler + loss
  6. Training loop (với tqdm progress bars)
  7. Validation mỗi epoch
  8. Save best model (theo val F1)
  9. Early stopping check
  10. Final test evaluation
  11. Generate plots
- **Windows-friendly**: Multiprocessing guard

### 7. **README.md** (7.2 KB)
- **Chức năng**: Tài liệu đầy đủ
- **Nội dung**:
  - Giới thiệu features
  - Cấu trúc dự án
  - Hướng dẫn sử dụng (Quick Start)
  - Giải thích tất cả CLI arguments
  - Output files
  - Tips & best practices
  - Requirements

### 8. **example_commands.py** (4.4 KB)
- **Chức năng**: Tập hợp các lệnh mẫu
- **Scenarios**:
  - Quick test (2 epochs)
  - Full training (CrossEntropy)
  - Full training (Focal Loss)
  - Transfer learning (frozen backbone)
  - CPU-only training
  - Hyperparameter experiments
  - Recommended pipeline (3 bước)

## 🚀 Lệnh Chạy Mẫu

### Lệnh Cơ Bản (Kiểm tra nhanh - 2 epochs)
```powershell
python train.py `
  --epochs 2 `
  --batch_size 16 `
  --lr 0.001 `
  --loss cross_entropy `
  --num_workers 0 `
  --output_dir "./test_outputs"
```

### Lệnh Đầy Đủ (Focal Loss - Khuyến nghị cho Imbalanced Data)
```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --epochs 100 `
  --batch_size 32 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss focal_loss `
  --focal_gamma 2.0 `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --num_workers 4 `
  --output_dir "./outputs_focal"
```

## 🎯 Tính Năng Nổi Bật

### Xử Lý Imbalanced Data (3 cấp độ)
1. **Class Weighting**: Tự động tính từ train set
2. **Focal Loss**: Loss function chuyên cho imbalance
3. **WeightedRandomSampler**: Oversample lớp thiểu số

### Augmentation Mạnh
- **RandAugment**: magnitude=9, num_ops=2
- **Geometric**: Rotation, Flip, ResizedCrop
- **Color**: ColorJitter
- **Normalization**: ImageNet mean/std

### Metrics Toàn Diện
- 4 metrics chính: Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
- Tracking cho cả train và validation
- Best model selection dựa trên **val F1-score**

### Visualization
- **Loss curves**: Train vs Val
- **F1 curves**: Train vs Val
- **Confusion Matrix**: Normalized heatmap (test set)

## 💡 Quy Trình Khuyến Nghị

### Bước 1: Validation Nhanh
```powershell
python train.py --epochs 2 --batch_size 16 --num_workers 0
```
→ Kiểm tra mọi thứ hoạt động, không có lỗi

### Bước 2: Tìm Loss Function Tốt Nhất
```powershell
# Thử CrossEntropy
python train.py --loss cross_entropy --epochs 20 --output_dir "./exp_ce"

# Thử Focal Loss
python train.py --loss focal_loss --epochs 20 --output_dir "./exp_focal"
```
→ So sánh val F1, chọn loss tốt hơn

### Bước 3: Full Training
```powershell
python train.py `
  --epochs 100 `
  --batch_size 32 `
  --lr 0.0001 `
  --loss focal_loss `
  --optimizer adamw `
  --weight_decay 0.01 `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --output_dir "./final_model"
```
→ Train đầy đủ với config tốt nhất

## 📊 Kết Quả Sau Training

File `outputs/training_history.json`:
```json
{
  "train_loss": [2.5, 2.0, 1.5, ...],
  "val_loss": [2.6, 2.1, 1.7, ...],
  "train_f1": [0.5, 0.6, 0.7, ...],
  "val_f1": [0.48, 0.58, 0.68, ...],
  ...
}
```

File `outputs/training.log` (ví dụ):
```
2026-02-05 14:00:00 - INFO - Starting ViT training pipeline...
2026-02-05 14:00:10 - INFO - Classes: ['Healthy Leaf', 'Insect_Pest_Disease', ...]
2026-02-05 14:00:10 - INFO - Class weights: [1.00, 1.25, 1.67, 2.50, 5.00, 3.33]
...
2026-02-05 14:30:00 - INFO - Epoch 1/100 | Train Loss: 2.1234 | Val Loss: 2.2345
2026-02-05 14:30:00 - INFO - Train F1: 0.4567 | Val F1: 0.4321
...
```

## ✅ Đã Hoàn Thành

- [x] 6 module Python modular (config, dataset, model, loss, utils, train)
- [x] Auto class weight calculation
- [x] Strong augmentation (RandAugment)
- [x] Dual loss support (CrossEntropy + Focal Loss)
- [x] 4 metrics tracking
- [x] Early stopping
- [x] Visualization (3 plots)
- [x] Windows compatibility
- [x] Comprehensive documentation
- [x] Example commands

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

Cài đặt:
```powershell
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
```
