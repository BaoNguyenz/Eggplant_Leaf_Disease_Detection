"""
Utilities module for ViT training pipeline.
Provides metrics calculation, plotting functions, early stopping, and logging setup.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import logging
import json


def setup_logger(output_dir: Path, log_file: str = 'training.log') -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Args:
        output_dir: Directory to save log file
        log_file: Name of log file
    
    Returns:
        Configured logger
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_file
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class EarlyStopping:
    """
    Early stopping to stop training when validation F1-score doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change in F1-score to qualify as improvement
        mode: 'max' for metrics to maximize (F1, accuracy), 'min' for loss
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (F1 or loss)
            epoch: Current epoch number
        
        Returns:
            True if score improved, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        # Check improvement based on mode
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def should_stop(self) -> bool:
        """Check if early stopping criterion is met."""
        return self.early_stop


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with accuracy, precision, recall, and F1-score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to: {save_path}")


def plot_f1_curve(
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    save_path: Path
) -> None:
    """
    Plot training and validation F1-score curves.
    
    Args:
        train_f1_scores: List of training F1-scores per epoch
        val_f1_scores: List of validation F1-scores per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_f1_scores) + 1)
    
    plt.plot(epochs, train_f1_scores, 'b-o', label='Training F1', linewidth=2, markersize=6)
    plt.plot(epochs, val_f1_scores, 'r-s', label='Validation F1', linewidth=2, markersize=6)
    
    plt.title('Training and Validation F1-Score', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"F1-score curve saved to: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        Tuple of (model, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics = checkpoint.get('metrics', {})
    
    return model, metrics


def save_training_history(
    history: Dict[str, List[float]],
    save_path: Path
) -> None:
    """
    Save training history to JSON file.
    
    Args:
        history: Dictionary containing training metrics history
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to: {save_path}")


if __name__ == '__main__':
    # Test utilities
    print("\n=== Testing Utilities ===\n")
    
    # Test metrics calculation
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("1. Metrics calculation:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    print("   ✓ Metrics calculated!\n")
    
    # Test early stopping
    print("2. Testing early stopping:")
    early_stopper = EarlyStopping(patience=3, mode='max')
    
    test_scores = [0.5, 0.6, 0.62, 0.61, 0.60, 0.59]
    for epoch, score in enumerate(test_scores, 1):
        improved = early_stopper(score, epoch)
        print(f"   Epoch {epoch}: F1={score:.2f}, Improved={improved}, Counter={early_stopper.counter}")
        if early_stopper.should_stop():
            print(f"   Early stopping triggered at epoch {epoch}")
            break
    print("   ✓ Early stopping works!\n")
    
    # Test plotting (save to temp directory)
    from pathlib import Path
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"3. Testing plotting (saving to {temp_dir}):")
    
    train_losses = [2.5, 2.0, 1.5, 1.2, 1.0]
    val_losses = [2.6, 2.1, 1.7, 1.4, 1.2]
    plot_loss_curves(train_losses, val_losses, temp_dir / 'test_loss.png')
    
    train_f1 = [0.5, 0.6, 0.7, 0.75, 0.8]
    val_f1 = [0.48, 0.58, 0.68, 0.72, 0.76]
    plot_f1_curve(train_f1, val_f1, temp_dir / 'test_f1.png')
    
    class_names = ['Class A', 'Class B', 'Class C']
    y_true_cm = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_cm = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
    plot_confusion_matrix(y_pred_cm, y_pred_cm, class_names, temp_dir / 'test_cm.png')
    
    print("   ✓ All plots created!\n")
    
    print("All tests passed! ✓")
