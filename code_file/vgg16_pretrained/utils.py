"""
Utility functions for VGG16 Eggplant Leaf Disease Detection.
Contains helper functions for class weights, metrics computation, and directory handling.
"""

import os
from pathlib import Path
from collections import Counter
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_class_weights(dataset) -> Dict[int, float]:
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        dataset: Dataset object with 'labels' attribute or list of labels.
        
    Returns:
        Dictionary mapping class index to weight.
    """
    # Extract labels from dataset
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        # Try to iterate through dataset to get labels
        labels = [label for _, label in dataset]
    
    # Count occurrences of each class
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    # Calculate weights: weight = total_samples / (num_classes * count_per_class)
    class_weights = {}
    for class_idx, count in label_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    return class_weights


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred: EvalPrediction object or tuple of (predictions, labels) from Trainer.
        
    Returns:
        Dictionary containing Accuracy, Precision, Recall, and F1-Score.
        All metrics use weighted average for multi-class classification.
    """
    # Handle both tuple and EvalPrediction object
    if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    
    # Ensure numpy arrays
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Handle logits (if predictions are 2D, take argmax)
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Ensure same length (truncate to minimum if mismatch)
    min_len = min(len(predictions), len(labels))
    if len(predictions) != len(labels):
        print(f"[WARNING] Length mismatch: predictions={len(predictions)}, labels={len(labels)}. Using first {min_len} samples.")
        predictions = predictions[:min_len]
        labels = labels[:min_len]
    
    # Compute metrics with weighted average for imbalanced data
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def check_dir(path: str) -> Path:
    """
    Check if directory exists and create it if not.
    
    Args:
        path: Path to the directory (string or Path object).
        
    Returns:
        Path object of the directory.
    """
    dir_path = Path(path)
    
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created directory: {dir_path}")
    else:
        print(f"[INFO] Directory already exists: {dir_path}")
    
    return dir_path


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
