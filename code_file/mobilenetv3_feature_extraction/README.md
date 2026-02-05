# MobileNetV3 Feature Extraction Pipeline

**Deep Learning Feature Extraction System for Eggplant Leaf Disease Detection**

## 📋 Overview

This modular Python system extracts **576-dimensional feature vectors** from eggplant leaf images using a **MobileNetV3 Small** model with **custom-trained weights**. Optimized for **Windows 10/11** with robust path handling and UTF-8 encoding support.

## 🏗️ Architecture

```
mobilenetv3_feature_extraction/
│
├── config.py              # Configuration management (paths, parameters, device)
├── dataset.py             # Custom Dataset class with ImageNet preprocessing
├── model.py               # MobileNetV3Extractor with custom weights loading
├── utils.py               # Helper functions (CSV I/O, validation)
├── extract.py             # Main execution script (Entry point)
├── README.md              # This file
│
├── mobilenetv3_output/    # Model weights directory
│   └── best_model.pth     # Custom trained weights (576-dim features)
│
└── extracted_features/    # Output directory (auto-created)
    └── mobilenetv3_features.csv
```

## 🔧 Technical Specifications

### Model Architecture
- **Base Model**: MobileNetV3 Small
- **Input Size**: 224 × 224 pixels
- **Preprocessing**: ImageNet normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])
- **Feature Extraction**:
  - Backbone: `features` (convolutional layers)
  - Pooling: `avgpool` (AdaptiveAvgPool2d)
  - **Classifier**: Removed entirely
- **Output Dimension**: **576** (flattened after avgpool)

### Key Features
- ✅ **Modular Design**: 5 separate modules for maintainability
- ✅ **Custom Weights Support**: Load `.pth` checkpoints from training
- ✅ **Windows Optimized**: `pathlib` for all path operations
- ✅ **Robust Error Handling**: DataParallel prefix removal, strict=False loading
- ✅ **UTF-8 CSV Export**: Windows-compatible encoding
- ✅ **Batch Processing**: Configurable batch size and multi-worker support
- ✅ **Type Hints**: Full type annotations for clarity

## 📦 Requirements

```bash
# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=9.0.0
```

**Installation:**
```powershell
pip install torch torchvision numpy pillow
```

## 🚀 Usage

### 1. Basic Execution (Default Paths)

```powershell
# Navigate to project directory
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction"

# Run extraction with default config
python extract.py
```

### 2. Custom Paths

```powershell
# Specify custom data directory
python extract.py --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images"

# Specify custom output directory
python extract.py --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\extracted_features"

# Specify custom weights
python extract.py --weight_path "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\mobilenetv3_output\best_model.pth"
```

### 3. Advanced Options

```powershell
# Adjust batch size and workers
python extract.py --batch_size 64 --num_workers 8

# Custom output filename
python extract.py --output_name "my_features.csv"

# Combine multiple options
python extract.py --data_dir "path\to\images" --batch_size 128 --num_workers 0
```

### 4. Full Command Example

```powershell
python extract.py `
    --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
    --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\extracted_features" `
    --weight_path "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\mobilenetv3_output\best_model.pth" `
    --batch_size 32 `
    --num_workers 4
```

## 📊 Output Format

### CSV Structure
```csv
filename,feature_0,feature_1,feature_2,...,feature_575
image_001.jpg,0.1234,0.5678,0.9012,...,0.3456
image_002.jpg,0.2345,0.6789,0.0123,...,0.4567
...
```

- **Column 1**: `filename` (original image filename)
- **Columns 2-577**: `feature_0` to `feature_575` (576 float values)

### Example Output
```
FEATURE EXTRACTION SUMMARY
======================================================================
Total Images Processed : 1500
Feature Dimension      : 576
Output File            : E:\...\extracted_features\mobilenetv3_features.csv
Time Elapsed           : 45.32 seconds
Processing Speed       : 33.10 images/sec
======================================================================
```

## 🧪 Testing Individual Modules

### Test Configuration
```powershell
python config.py
```

### Test Dataset
```powershell
python dataset.py
```

### Test Model
```powershell
python model.py
```

### Test Utilities
```powershell
python utils.py
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. `FileNotFoundError: Weight file not found`
**Solution:** Ensure `best_model.pth` exists at:
```
E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\mobilenetv3_output\best_model.pth
```

#### 2. `RuntimeError: CUDA out of memory`
**Solution:** Reduce batch size:
```powershell
python extract.py --batch_size 16
```

#### 3. DataLoader Hanging on Windows
**Solution:** Set `num_workers=0`:
```powershell
python extract.py --num_workers 0
```

#### 4. `UnicodeDecodeError` in CSV
**Solution:** Already handled! This system uses UTF-8 encoding by default.

#### 5. Missing Keys in State Dict
**Issue:** Checkpoint has `module.` prefix (DataParallel training)  
**Solution:** Automatically handled by `_remove_module_prefix()` in `model.py`

## 📁 Directory Structure Details

```
mobilenetv3_feature_extraction/
│
├── Source Code (5 files)
│   ├── config.py          # 150 lines - Configuration
│   ├── dataset.py         # 180 lines - Data loading
│   ├── model.py           # 200 lines - MobileNetV3 + weights
│   ├── utils.py           # 160 lines - Utilities
│   └── extract.py         # 220 lines - Main script
│
├── Inputs
│   └── mobilenetv3_output/
│       └── best_model.pth # Custom weights (~10 MB)
│
└── Outputs
    └── extracted_features/
        └── mobilenetv3_features.csv # 576-dim features
```

## 🎯 Feature Extraction Pipeline

```mermaid
graph LR
    A[Input Images] --> B[Dataset Loader]
    B --> C[Preprocessing<br/>224x224, Normalize]
    C --> D[MobileNetV3<br/>features]
    D --> E[avgpool<br/>576x1x1]
    E --> F[Flatten<br/>576-dim]
    F --> G[CSV Output]
```

## 🔍 Tensor Flow Visualization

```python
Input:        [B, 3, 224, 224]  # RGB images
              ↓
features:     [B, 576, 7, 7]    # Convolutional backbone
              ↓
avgpool:      [B, 576, 1, 1]    # Global Average Pooling
              ↓
flatten:      [B, 576]          # Final feature vector
```

## 📝 Code Quality Standards

- ✅ **PEP 8 Compliance**: Proper naming, spacing, docstrings
- ✅ **Type Hints**: All functions fully annotated
- ✅ **Extensive Comments**: Logic and tensor shapes explained
- ✅ **Error Handling**: Try-except blocks with meaningful messages
- ✅ **Logging**: Progress updates and validation

## 🔗 Integration with ML Pipeline

### Downstream Usage Examples

#### 1. Load Features for Classification
```python
import pandas as pd

# Load features
df = pd.read_csv('extracted_features/mobilenetv3_features.csv')
X = df.iloc[:, 1:].values  # Features (576 columns)
filenames = df['filename'].values

# Train classifier (SVM, Random Forest, etc.)
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X, y_train)
```

#### 2. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X)  # 576 -> 100 dimensions
```

## 🧑‍💻 Development Notes

### Custom Weights Loading Logic
The system handles three checkpoint formats:
1. `{'model_state_dict': ...}`
2. `{'state_dict': ...}`
3. Direct state dictionary

It also:
- Removes `module.` prefix (DataParallel)
- Filters only `features.*` and `avgpool.*` keys
- Uses `strict=False` to allow missing classifier keys

### Windows Path Handling
All paths use `pathlib.Path` to:
- Avoid `MAX_PATH` limitations
- Handle backslashes automatically
- Enable cross-platform compatibility

## 📞 Support

For issues related to:
- **Model architecture**: Check `model.py` tensor flow comments
- **Data loading**: Verify dataset structure in `dataset.py`
- **Path errors**: Ensure all paths use `pathlib.Path`
- **CSV encoding**: UTF-8 is enforced in `utils.py`

## 📄 License

This code is part of the Eggplant Leaf Disease Detection research project.

---

**Created by**: AI Engineering Team  
**Date**: February 4, 2026  
**Version**: 1.0.0
