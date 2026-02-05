# ViT Training Pipeline - Eggplant Leaf Disease Detection

Modular PyTorch training pipeline for Vision Transformer (ViT-B/16) to classify eggplant leaf diseases with imbalanced data handling.

## 📁 Project Structure

```
vit_pretrained/
├── config.py           # CLI argument parsing and configuration
├── dataset.py          # Data loading, augmentation, and class weighting
├── model.py            # ViT-B/16 model with ImageNet weights
├── loss.py             # CrossEntropyLoss and Focal Loss
├── utils.py            # Metrics, plotting, early stopping
├── train.py            # Main training script
└── README.md           # This file
```

## 🎯 Features

### Class Imbalance Handling
- **Auto class weight calculation** from training set distribution
- **WeightedRandomSampler** option for oversampling minority classes
- **Weighted loss functions** (both CrossEntropy and Focal Loss)

### Data Augmentation
- **Strong augmentation** using `torchvision.transforms.v2`:
  - RandAugment (num_ops=2, magnitude=9)
  - RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
  - ColorJitter, RandomRotation
- **ImageNet normalization** for transfer learning

### Model Architecture
- **ViT-B/16** with IMAGENET1K_V1 pretrained weights
- Optional **backbone freezing** for feature extraction mode
- 768-dimensional embeddings → 6 disease classes

### Training Features
- **4 key metrics**: Accuracy, Precision, Recall, F1-Score (weighted)
- **Stratified data splitting**: 80/10/10 train/val/test
- **Early stopping** based on validation F1-score
- **Learning rate scheduling**: ReduceLROnPlateau
- **Comprehensive logging** with file and console output

### Visualization
- Training & validation loss curves
- F1-score progression
- Normalized confusion matrix (heatmap)

## 🚀 Quick Start

### Basic Training

```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.0001 `
  --loss cross_entropy `
  --output_dir "./outputs"
```

### Advanced Training with Focal Loss

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

### Fine-tuning Only Classification Head

```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --epochs 30 `
  --batch_size 64 `
  --lr 0.001 `
  --freeze_backbone `
  --output_dir "./outputs_frozen"
```

## 📊 Command-Line Arguments

### Data Parameters
- `--data_dir`: Path to dataset (default: dataset path)
- `--train_split`: Training proportion (default: 0.8)
- `--val_split`: Validation proportion (default: 0.1)

### Training Parameters
- `--epochs`: Number of epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: L2 regularization (default: 0.0)

### Model Parameters
- `--num_classes`: Number of classes (default: 6)
- `--freeze_backbone`: Freeze ViT encoder (flag)

### Loss Function
- `--loss`: Loss type - `cross_entropy` or `focal_loss` (default: cross_entropy)
- `--focal_gamma`: Focal Loss gamma parameter (default: 2.0)

### Optimizer
- `--optimizer`: Optimizer - `adam` or `adamw` (default: adamw)
- `--use_scheduler`: Enable ReduceLROnPlateau (flag)

### Early Stopping
- `--early_stopping`: Patience in epochs, 0=disabled (default: 10)

### System Parameters
- `--num_workers`: Data loading workers (default: 4, set to 0 if Windows issues)
- `--device`: Device - `cuda`, `cpu`, or `auto` (default: auto)
- `--seed`: Random seed (default: 42)

### Output
- `--output_dir`: Output directory (default: ./outputs)

### Sampling
- `--use_weighted_sampler`: Use WeightedRandomSampler (flag)

## 📈 Output Files

After training, the `output_dir` will contain:

```
outputs/
├── best_model.pth              # Best model checkpoint (by val F1)
├── training_history.json       # All metrics per epoch
├── loss_curves.png             # Train/val loss plot
├── f1_curve.png                # Train/val F1-score plot
├── confusion_matrix.png        # Test set confusion matrix
└── training.log                # Complete training log
```

## 🧪 Testing Individual Modules

Each module can be tested independently:

```powershell
# Test configuration
python config.py --help

# Test dataset loading
python dataset.py

# Test model creation
python model.py

# Test loss functions
python loss.py

# Test utilities
python utils.py
```

## 💡 Tips for Best Results

### For Imbalanced Data
1. **Try Focal Loss first**: Better than CrossEntropy for severe imbalance
   ```powershell
   --loss focal_loss --focal_gamma 2.0
   ```

2. **Enable weighted sampler**: Oversamples minority classes
   ```powershell
   --use_weighted_sampler
   ```

3. **Both methods together**: Maximum imbalance handling
   ```powershell
   --loss focal_loss --use_weighted_sampler
   ```

### For Better Generalization
1. **Use AdamW with weight decay**:
   ```powershell
   --optimizer adamw --weight_decay 0.01
   ```

2. **Enable learning rate scheduling**:
   ```powershell
   --use_scheduler
   ```

3. **Adjust early stopping patience**:
   ```powershell
   --early_stopping 15  # Wait longer for improvements
   ```

### For Faster Training
1. **Freeze backbone initially**:
   ```powershell
   --freeze_backbone --lr 0.001 --epochs 20
   ```

2. **Then fine-tune full model**:
   ```powershell
   --lr 0.0001 --epochs 50  # Load from previous best_model.pth
   ```

### Windows-Specific
1. **If multiprocessing errors occur**:
   ```powershell
   --num_workers 0
   ```

2. **For smaller GPUs/RAM**:
   ```powershell
   --batch_size 16  # Reduce from default 32
   ```

## 📋 Requirements

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

Install with:
```powershell
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
```

## 🎓 Dataset Structure

Expected directory structure:
```
Classified Images/
├── Healthy Leaf/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Insect_Pest_Disease/
│   └── ...
├── Leaf_Spot_Disease/
│   └── ...
├── Mosaic_Virus_Disease/
│   └── ...
├── White_Mold_Disease/
│   └── ...
└── Wilt_Disease/
    └── ...
```

## 📝 License

This code is provided as-is for educational and research purposes.

## 👤 Author

Senior AI Engineer & Python Software Architect
Date: 2026-02-05
