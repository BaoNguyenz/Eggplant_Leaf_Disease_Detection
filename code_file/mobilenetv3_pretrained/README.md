# MobileNetV3 Small Training Pipeline
**Eggplant Leaf Disease Detection - Multi-class Image Classification**

## 📋 Overview
This project implements a complete training pipeline for **MobileNetV3 Small** (pretrained on ImageNet) to detect diseases on eggplant leaves. The codebase is optimized for Windows and designed with modularity, customization, and reproducibility in mind.

## 🗂️ Project Structure
```
mobilenetv3_pretrained/
├── utils.py           # Utility functions (metrics, visualization, Focal Loss)
├── data_setup.py      # Dataset loading and augmentation pipeline
├── model_setup.py     # MobileNetV3 model configuration
├── train.py           # Main training script
└── README.md          # This file
```

## ✨ Key Features
- ✅ **MobileNetV3 Small** with ImageNet pretrained weights
- ✅ **224×224 input resolution** (optimized for mobile/embedded devices)
- ✅ **Class weight handling** for imbalanced datasets
- ✅ **Dual loss functions**: Cross Entropy & Focal Loss
- ✅ **Rich data augmentation**: RandomResizedCrop, ColorJitter, GaussianBlur, Gaussian Noise, Rotation
- ✅ **80/10/10 train/val/test split** with reproducible seeding
- ✅ **Early stopping** based on validation F1-score
- ✅ **Comprehensive visualization**: Loss curves, F1 curves, Confusion matrix
- ✅ **Detailed logging**: CSV training log, Classification report
- ✅ **Full argument parsing** for flexible experimentation

## 🚀 Quick Start

### 1. Basic Training (Default Settings)
```bash
python train.py
```
This will use the default paths and hyperparameters defined in the script.

### 2. Custom Training
```bash
python train.py \
    --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" \
    --output_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_pretrained" \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --num_workers 4 \
    --loss_type focal \
    --patience 10 \
    --seed 42
```

### 3. Quick Experiment (5 epochs for testing)
```bash
python train.py --epochs 5 --batch_size 16
```

## 🎛️ Command-Line Arguments

| Argument        | Type   | Default                                                      | Description                              |
|-----------------|--------|--------------------------------------------------------------|------------------------------------------|
| `--data_dir`    | str    | `E:\...\Classified Images`                                   | Path to dataset directory                |
| `--output_dir`  | str    | `E:\...\mobilenetv3_pretrained`                              | Path to save outputs                     |
| `--epochs`      | int    | 50                                                           | Number of training epochs                |
| `--batch_size`  | int    | 32                                                           | Batch size                               |
| `--lr`          | float  | 0.001                                                        | Learning rate                            |
| `--num_workers` | int    | 4                                                            | Number of data loading workers           |
| `--seed`        | int    | 42                                                           | Random seed for reproducibility          |
| `--loss_type`   | str    | `cross_entropy`                                              | Loss function (`cross_entropy`, `focal`) |
| `--patience`    | int    | 10                                                           | Early stopping patience (epochs)         |

## 📊 Output Files
After training, the following files will be saved in `output_dir`:

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `best_model.pth`              | Best model checkpoint (based on Val F1-score)   |
| `training_log.csv`            | Epoch-by-epoch metrics (loss, F1)               |
| `training_curves.png`         | Loss & F1-score curves (train & validation)     |
| `confusion_matrix.png`        | Heatmap of test set predictions                 |
| `classification_report.txt`   | Detailed per-class metrics                      |

## 🔬 Data Augmentation Pipeline

### Training Transforms
```python
RandomResizedCrop(224, scale=(0.8, 1.0))
RandomHorizontalFlip(p=0.5)
RandomRotation(15)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
ToTensor()
AddGaussianNoise(mean=0.0, std=0.05)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Validation/Test Transforms
```python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 🏗️ Model Architecture
- **Backbone**: MobileNetV3 Small (pretrained on ImageNet)
- **Input Size**: 224×224×3
- **Feature Extractor**: MobileNetV3 features (all layers trainable by default)
- **Classifier**: Modified final linear layer to match `num_classes`
- **Output**: Softmax probabilities for each class

## 📈 Training Process
1. **Initialization**: Set random seed, create dataloaders, initialize model
2. **Training Loop**: 
   - Train on 80% of data with augmentation
   - Validate on 10% of data (no augmentation)
   - Track loss and F1-score
   - Save best model based on validation F1
   - Early stopping if no improvement for `patience` epochs
3. **Post-Training**:
   - Load best model weights
   - Evaluate on 10% test set
   - Generate confusion matrix, classification report
   - Save all visualizations and logs

## 💡 Tips & Best Practices

### For Imbalanced Datasets
```bash
python train.py --loss_type focal
```
Focal Loss automatically down-weights easy examples and focuses on hard cases.

### For Limited GPU Memory
```bash
python train.py --batch_size 16 --num_workers 2
```

### For Faster Iteration (Prototyping)
```bash
python train.py --epochs 10 --patience 5
```

### For Maximum Reproducibility
```bash
python train.py --seed 42 --num_workers 0
```
Note: `num_workers=0` ensures deterministic behavior but may be slower.

## 🛠️ Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

## 📝 Module Descriptions

### `utils.py`
- `set_seed()`: Ensures reproducibility
- `get_class_weights()`: Calculates inverse frequency weights
- `compute_metrics()`: Computes accuracy, precision, recall, F1
- `save_training_log()`: Exports metrics to CSV
- `save_loss_f1_curves()`: Generates training curves
- `save_confusion_matrix()`: Creates heatmap visualization
- `FocalLoss`: Custom loss for handling class imbalance

### `data_setup.py`
- `AddGaussianNoise`: Custom transform for data augmentation
- `create_dataloaders()`: Handles dataset splitting and augmentation

### `model_setup.py`
- `create_mobilenetv3_model()`: Loads pretrained model and modifies classifier
- `count_parameters()`: Reports trainable/total parameters

### `train.py`
- Main training loop with progress bars
- Argument parsing
- Checkpoint saving and loading
- Post-training evaluation and visualization

## 📧 Support
For issues or questions related to this implementation, please refer to the official PyTorch documentation:
- [MobileNetV3 Documentation](https://pytorch.org/vision/stable/models.html#mobilenetv3)
- [PyTorch Transforms](https://pytorch.org/vision/stable/transforms.html)

---

**Author**: AI-Generated Training Pipeline  
**Framework**: PyTorch  
**License**: Use for educational and research purposes
