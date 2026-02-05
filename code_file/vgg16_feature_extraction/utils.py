"""
Utility Module
Helper functions for feature extraction pipeline.
"""
import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json


def save_features_to_csv(
    features: np.ndarray,
    labels: List[str],
    image_paths: List,  # Can be List[Path] or List[str]
    output_path: Path,
    encoding: str = 'utf-8'
) -> None:
    """
    Save extracted features to CSV file with UTF-8 encoding.
    
    Args:
        features: Numpy array of shape (N, 512) containing feature vectors
        labels: List of class labels for each sample
        image_paths: List of image paths
        output_path: Path to output CSV file
        encoding: Character encoding for CSV file
        
    Raises:
        IOError: If file writing fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding=encoding) as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
            header = ['image_path', 'label'] + feature_columns
            writer.writerow(header)
            
            # Write data rows
            for i, (feat, label, img_path) in enumerate(zip(features, labels, image_paths)):
                row = [str(img_path), label] + feat.tolist()
                writer.writerow(row)
        
        print(f"✓ Features saved to: {output_path}")
        print(f"  - Total samples: {len(features)}")
        print(f"  - Feature dimension: {features.shape[1]}")
    
    except Exception as e:
        raise IOError(f"Failed to save CSV file: {str(e)}")


def save_metadata(
    output_dir: Path,
    config: Dict,
    num_samples: int,
    num_classes: int,
    class_names: List[str],
    extraction_time: float
) -> None:
    """
    Save extraction metadata to JSON file.
    
    Args:
        output_dir: Directory to save metadata
        config: Configuration dictionary
        num_samples: Total number of samples processed
        num_classes: Number of classes
        class_names: List of class names
        extraction_time: Time taken for extraction (seconds)
    """
    metadata = {
        'extraction_date': datetime.now().isoformat(),
        'model': config.get('model_name', 'VGG16'),
        'feature_dim': config.get('feature_dim', 512),
        'num_samples': num_samples,
        'num_classes': num_classes,
        'class_names': class_names,
        'extraction_time_seconds': round(extraction_time, 2),
        'image_size': config.get('img_size', [224, 224]),
        'batch_size': config.get('batch_size', 32)
    }
    
    metadata_path = output_dir / 'extraction_metadata.json'
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Metadata saved to: {metadata_path}")


def print_extraction_summary(
    num_samples: int,
    num_classes: int,
    class_names: List[str],
    feature_dim: int,
    extraction_time: float
) -> None:
    """
    Print summary of feature extraction.
    
    Args:
        num_samples: Total samples processed
        num_classes: Number of classes
        class_names: List of class names
        feature_dim: Feature vector dimension
        extraction_time: Time taken (seconds)
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total samples:       {num_samples:,}")
    print(f"Number of classes:   {num_classes}")
    print(f"Feature dimension:   {feature_dim}")
    print(f"Extraction time:     {extraction_time:.2f} seconds")
    print(f"Throughput:          {num_samples/extraction_time:.2f} images/sec")
    print("\nClass distribution:")
    for i, class_name in enumerate(class_names, 1):
        print(f"  {i}. {class_name}")
    print("="*60 + "\n")


def validate_gpu_availability() -> str:
    """
    Check GPU availability and return device string.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        device = 'cpu'
        print("⚠ GPU not available, using CPU")
    
    return device


def create_output_filename(prefix: str = 'vgg16_features', extension: str = 'csv') -> str:
    """
    Create timestamped output filename.
    
    Args:
        prefix: Filename prefix
        extension: File extension without dot
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"


def normalize_path_for_windows(path: Path) -> Path:
    """
    Normalize path for Windows compatibility.
    Resolves relative paths and handles long path issues.
    
    Args:
        path: Input path
        
    Returns:
        Resolved absolute path
    """
    return path.resolve()
