"""
Utility functions for ViT feature extraction pipeline.
Handles device setup, CSV saving, and directory management.
"""
from pathlib import Path
from typing import List
import csv

import torch
import numpy as np


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get computing device (CUDA or CPU).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device instance
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("[INFO] Using CPU")
    
    return device


def create_output_dir(output_dir: Path) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory ready: {output_dir}")


def save_features_to_csv(
    features: np.ndarray,
    image_paths: List[str],
    labels: List[str],
    output_path: Path,
    feature_dim: int = 768
) -> None:
    """
    Save extracted features to CSV file.
    
    CSV Format:
        image_path, class_name, feature_1, feature_2, ..., feature_768
    
    Args:
        features: Feature array of shape [num_samples, feature_dim]
        image_paths: List of image file paths
        labels: List of class labels
        output_path: Path to output CSV file
        feature_dim: Dimension of feature vectors
    """
    output_path = Path(output_path)
    
    # Validate inputs
    assert features.shape[0] == len(image_paths) == len(labels), \
        "Mismatch between features, paths, and labels lengths"
    assert features.shape[1] == feature_dim, \
        f"Expected feature dimension {feature_dim}, got {features.shape[1]}"
    
    # Create CSV header
    header = ['image_path', 'class_name'] + [f'feature_{i+1}' for i in range(feature_dim)]
    
    # Write to CSV with UTF-8 encoding (Windows compatible)
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(header)
            
            # Write data rows
            for img_path, label, feature_vec in zip(image_paths, labels, features):
                row = [img_path, label] + feature_vec.tolist()
                writer.writerow(row)
        
        print(f"[SUCCESS] Features saved to: {output_path}")
        print(f"[INFO] Total samples: {len(image_paths)}")
        print(f"[INFO] Feature dimension: {feature_dim}")
        
    except Exception as e:
        raise RuntimeError(f"Error saving CSV file: {e}")


def print_extraction_summary(
    num_images: int,
    num_classes: int,
    feature_dim: int,
    output_csv: Path,
    device: torch.device
) -> None:
    """
    Print summary of feature extraction process.
    
    Args:
        num_images: Total number of images processed
        num_classes: Number of unique classes
        feature_dim: Feature vector dimension
        output_csv: Path to output CSV file
        device: Device used for extraction
    """
    print("\n" + "="*70)
    print(" " * 20 + "EXTRACTION SUMMARY")
    print("="*70)
    print(f"{'Total Images Processed:':<30} {num_images:>10,}")
    print(f"{'Number of Classes:':<30} {num_classes:>10,}")
    print(f"{'Feature Dimension:':<30} {feature_dim:>10}")
    print(f"{'Device Used:':<30} {str(device):>10}")
    print(f"{'Output CSV:':<30}")
    print(f"  {output_csv}")
    print("="*70 + "\n")


def validate_paths(data_dir: Path, output_dir: Path) -> None:
    """
    Validate input and output paths.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")
    
    print(f"[INFO] Data directory validated: {data_dir}")
    print(f"[INFO] Output directory: {output_dir}")
