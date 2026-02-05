"""
Utility functions for EfficientNet-B7 training pipeline.
Includes class weights calculation, metrics, visualization, and Focal Loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory ensured: {path}")


def get_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        
    Returns:
        Tensor of class weights
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    num_classes = len(class_counts)
    
    # Calculate weights: inversely proportional to class frequency
    class_weights = total_samples / (num_classes * class_counts)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION & WEIGHTS")
    print("="*60)
    for idx, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"Class {idx}: {count:5d} samples | Weight: {weight:.4f}")
    print("="*60 + "\n")
    
    return torch.FloatTensor(class_weights)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-Score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing weighted metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def save_loss_f1_curves(history: dict, output_dir: Path) -> None:
    """
    Plot and save training/validation Loss and F1-Score curves.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1-Score
    axes[1].plot(history['train_f1'], label='Train F1', marker='o', linewidth=2)
    axes[1].plot(history['val_f1'], label='Val F1', marker='s', linewidth=2)
    axes[1].set_title('Training & Validation F1-Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1-Score (Weighted)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'loss_f1_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss & F1 curves saved: {save_path}")


def save_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: list, 
                         output_dir: Path) -> None:
    """
    Plot and save confusion matrix as heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples.
    
    Reference: Lin et al. - Focal Loss for Dense Object Detection
    """
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def save_classification_report(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               class_names: list, 
                               output_dir: Path) -> None:
    """
    Generate and save classification report to text file.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    check_dir(output_dir)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    save_path = output_dir / 'classification_report.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT - EfficientNet-B7\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Classification report saved: {save_path}")
    print("\n" + report)
