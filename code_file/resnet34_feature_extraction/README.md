# ResNet34 Feature Extraction

Hệ thống trích xuất đặc trưng sử dụng ResNet34 với trọng số tùy chỉnh (Custom Weights) cho bài toán phát hiện bệnh lá cà tím.

## 📁 Cấu trúc thư mục

```
resnet34_feature_extraction/
│
├── config.py              # Cấu hình đường dẫn và tham số
├── dataset.py             # Dataset class và transforms
├── model.py               # ResNet34 feature extractor
├── utils.py               # Các hàm tiện ích
├── extract.py             # Script chính (Entry point)
├── README.md              # File này
│
└── extracted_features/    # Thư mục output (tự động tạo)
    └── resnet34_features.csv
```

## 🎯 Tính năng

### Kiến trúc Modular
- ✅ **5 modules độc lập**: Dễ bảo trì và mở rộng
- ✅ **Type Hinting đầy đủ**: Tăng khả năng đọc code
- ✅ **PEP 8 compliant**: Clean Code chuẩn Python

### Tối ưu cho Windows
- ✅ **Pathlib 100%**: Xử lý đường dẫn an toàn, tránh lỗi `MAX_PATH`
- ✅ **UTF-8 encoding**: Hỗ trợ tiếng Việt trong CSV
- ✅ **Error handling**: Try-except khi load weights

### ResNet34 Custom Weights
- ✅ **Load từ .pth file**: Hỗ trợ nhiều format checkpoint
- ✅ **Handle key mismatch**: Tự động xử lý prefix `module.`
- ✅ **Strict=False option**: Linh hoạt với kiến trúc khác nhau
- ✅ **512-dimensional features**: Global Average Pooling output

### CLI Arguments
- ✅ **Flexible paths**: Override mọi đường dẫn qua command line
- ✅ **Batch processing**: Điều chỉnh batch size và num_workers
- ✅ **Custom output**: Tùy chỉnh tên file CSV output

## 📦 Dependencies

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.0.0
tqdm>=4.65.0
```

## 🚀 Cách sử dụng

### 1. Sử dụng đường dẫn mặc định

```powershell
# Chạy với cấu hình mặc định từ config.py
python extract.py
```

### 2. Chỉ định đường dẫn tùy chỉnh

```powershell
# Chỉ định custom data directory
python extract.py --data_dir "E:\Custom\Path\To\Images"

# Chỉ định output directory
python extract.py --output_dir "E:\Custom\Output\Path"

# Sử dụng weights khác
python extract.py --weight_path "E:\Custom\Weights\best_model.pth"
```

### 3. Điều chỉnh hiệu năng

```powershell
# Tăng batch size (nếu có GPU mạnh)
python extract.py --batch_size 64

# Giảm num_workers (nếu gặp lỗi trên Windows)
python extract.py --num_workers 0
```

### 4. Ví dụ đầy đủ với đường dẫn thực tế

```powershell
# Chạy với đường dẫn đầy đủ
python extract.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\resnet34_feature_extraction\extracted_features" `
  --weight_path "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\resnet34_feature_extract\Resnet34_output\best_model.pth" `
  --batch_size 32 `
  --num_workers 4
```

## 📊 Output Format

File CSV được tạo với cấu trúc:

```csv
filename,label,feature_0,feature_1,...,feature_511
image_001.jpg,healthy,0.123,-0.456,...,0.789
image_002.jpg,diseased,0.234,-0.567,...,0.890
...
```

- **filename**: Tên file ảnh
- **label**: Tên class (tên thư mục chứa ảnh)
- **feature_0 đến feature_511**: 512 giá trị đặc trưng

## 🔧 Troubleshooting

### Lỗi thiếu file weights

```
FileNotFoundError: Weight file not found: ...
```

**Giải pháp**: Kiểm tra đường dẫn trong `config.py` hoặc truyền `--weight_path` chính xác.

### Lỗi load weights không khớp kiến trúc

```
RuntimeError: Error(s) in loading state_dict...
```

**Giải pháp**: Code đã xử lý với `strict=False`. Kiểm tra log để xem missing/unexpected keys.

### Lỗi DataLoader trên Windows

```
BrokenPipeError or RuntimeError in DataLoader worker
```

**Giải pháp**: Đặt `--num_workers 0` để chạy trên main process.

### Lỗi out of memory (GPU)

```
RuntimeError: CUDA out of memory
```

**Giải pháp**: Giảm `--batch_size` xuống 16 hoặc 8.

## 🧪 Testing

Test từng module:

```powershell
# Test config
python config.py

# Test dataset
python dataset.py

# Test model
python model.py

# Test utils
python utils.py
```

## 📝 Notes

- **Checkpoint format**: Code tự động xử lý các format phổ biến:
  - Direct state_dict
  - `{'state_dict': ...}`
  - `{'model_state_dict': ...}`
  - Keys with `module.` prefix (từ DataParallel)

- **Image preprocessing**: 
  - Resize: (224, 224)
  - Normalize: ImageNet mean/std
  - Color mode: RGB

- **Feature extraction**:
  - Layer: Sau Global Average Pooling
  - Dimension: 512
  - No gradient computation

## 👤 Author

Kỹ sư AI - Eggplant Leaf Disease Detection Project

## 📄 License

Internal Research Project
