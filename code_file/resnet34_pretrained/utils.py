"""
Utility functions for eggplant leaf disease detection using ResNet34.
Includes visualization, metrics computation, and FocalLoss implementation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def set_seed(seed=42):
    """
    Set random seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def get_class_weights(dataset):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        dataset: PyTorch Dataset object with targets attribute
        
    Returns:
        torch.Tensor: Class weights normalized by inverse frequency
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    num_classes = len(class_counts)
    
    # Calculate inverse frequency weights
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"✓ Class weights calculated: {class_weights.numpy()}")
    return class_weights


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-Score (Weighted).
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def check_dir(path):
    """
    Safely create directory if it doesn't exist.
    
    Args:
        path (str or Path): Directory path to create
        
    Returns:
        Path: Path object of the created/existing directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_training_log(history, output_dir):
    """
    Save training history to CSV file.
    
    Args:
        history (dict): Dictionary containing training metrics with keys like
                       'train_loss', 'train_f1', 'val_loss', 'val_f1'
        output_dir (str or Path): Directory to save the log file
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    df = pd.DataFrame(history)
    csv_path = output_dir / 'training_log.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Training log saved to: {csv_path}")


def save_loss_f1_curves(history, output_dir):
    """
    Plot and save Loss and F1-Score curves during training.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'train_f1', 'val_loss', 'val_f1'
        output_dir (str or Path): Directory to save the plot
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=5)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # F1-Score curves
    ax2.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2, markersize=5)
    ax2.plot(epochs, history['val_f1'], 'r-s', label='Val F1', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation F1-Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {plot_path}")


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
        output_dir (str or Path): Directory to save the confusion matrix
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Class weights. Shape: [num_classes]
            gamma (float): Focusing parameter for modulating loss. Default: 2.0
            reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Logits from model. Shape: [batch_size, num_classes]
            targets (Tensor): Ground truth labels. Shape: [batch_size]
            
        Returns:
            Tensor: Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
