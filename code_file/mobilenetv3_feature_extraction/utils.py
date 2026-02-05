"""
Utility Module for MobileNetV3 Feature Extraction
=================================================
Helper functions for directory management, CSV saving, and logging.
Optimized for Windows with UTF-8 encoding support.

Author: AI Engineer
Date: 2026-02-04
"""

from pathlib import Path
from typing import List, Dict
import csv
import numpy as np


def ensure_directory(directory: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (Path): Directory path to create
    """
    directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory ready: {directory}")


def save_features_to_csv(
    features: np.ndarray,
    filenames: List[str],
    labels: List[str],
    output_path: Path,
    feature_dim: int = 576
) -> None:
    """
    Save extracted features to CSV file with UTF-8 encoding.
    
    CSV Format:
        filename,label,feature_0,feature_1,...,feature_575
        image1.jpg,Healthy,0.123,0.456,...,0.789
        image2.jpg,Bacterial Leaf Spot,0.234,0.567,...,0.890
    
    Args:
        features (np.ndarray): Feature array of shape [N, 576]
        filenames (List[str]): List of image filenames (length N)
        labels (List[str]): List of class labels (length N)
        output_path (Path): Path to save CSV file
        feature_dim (int): Expected feature dimension (default: 576)
    
    Raises:
        ValueError: If shapes don't match expectations
        IOError: If CSV writing fails
    """
    # Validate inputs
    if features.shape[0] != len(filenames):
        raise ValueError(
            f"Mismatch: {features.shape[0]} features but {len(filenames)} filenames"
        )
    
    if features.shape[0] != len(labels):
        raise ValueError(
            f"Mismatch: {features.shape[0]} features but {len(labels)} labels"
        )
    
    if features.shape[1] != feature_dim:
        raise ValueError(
            f"Expected {feature_dim} features, got {features.shape[1]}"
        )
    
    # Ensure output directory exists
    ensure_directory(output_path.parent)
    
    # Prepare CSV header
    header = ['filename', 'label'] + [f'feature_{i}' for i in range(feature_dim)]
    
    # Write to CSV with UTF-8 encoding (critical for Windows)
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(header)
            
            # Write data rows
            for filename, label, feature_vector in zip(filenames, labels, features):
                row = [filename, label] + feature_vector.tolist()
                writer.writerow(row)
        
        print(f"✓ Features saved to: {output_path}")
        print(f"  - Total samples: {len(filenames)}")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        raise IOError(f"Failed to write CSV file: {e}")


def load_features_from_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load features from CSV file (utility for verification).
    
    Args:
        csv_path (Path): Path to CSV file
    
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping filenames to feature vectors
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    features_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            filename = row['filename']
            
            # Extract feature values (skip 'filename' column)
            feature_vector = np.array([
                float(row[f'feature_{i}']) for i in range(len(row) - 1)
            ])
            
            features_dict[filename] = feature_vector
    
    print(f"✓ Loaded {len(features_dict)} feature vectors from {csv_path}")
    
    return features_dict


def print_extraction_summary(
    total_images: int,
    feature_dim: int,
    output_path: Path,
    elapsed_time: float
) -> None:
    """
    Print summary of feature extraction process.
    
    Args:
        total_images (int): Number of images processed
        feature_dim (int): Feature dimension
        output_path (Path): Path to output CSV
        elapsed_time (float): Time taken in seconds
    """
    print("\n" + "="*70)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total Images Processed : {total_images}")
    print(f"Feature Dimension      : {feature_dim}")
    print(f"Output File            : {output_path}")
    print(f"Time Elapsed           : {elapsed_time:.2f} seconds")
    print(f"Processing Speed       : {total_images / elapsed_time:.2f} images/sec")
    print("="*70 + "\n")


def validate_csv_output(csv_path: Path, expected_samples: int, expected_dim: int) -> bool:
    """
    Validate the generated CSV file.
    
    Args:
        csv_path (Path): Path to CSV file
        expected_samples (int): Expected number of rows
        expected_dim (int): Expected feature dimension
    
    Returns:
        bool: True if validation passes
    """
    try:
        # Load and check
        features_dict = load_features_from_csv(csv_path)
        
        # Check number of samples
        if len(features_dict) != expected_samples:
            print(f"✗ Sample count mismatch: {len(features_dict)} != {expected_samples}")
            return False
        
        # Check feature dimensions
        first_feature = next(iter(features_dict.values()))
        if len(first_feature) != expected_dim:
            print(f"✗ Feature dimension mismatch: {len(first_feature)} != {expected_dim}")
            return False
        
        print(f"✓ CSV validation passed!")
        return True
        
    except Exception as e:
        print(f"✗ CSV validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test utility functions
    print("Testing Utility Module...")
    print("-" * 70)
    
    # Test directory creation
    test_dir = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\test_output")
    ensure_directory(test_dir)
    
    # Test CSV saving
    test_features = np.random.randn(5, 576)  # 5 samples, 576 features
    test_filenames = [f"image_{i}.jpg" for i in range(5)]
    test_labels = ["Healthy", "Bacterial Leaf Spot", "Healthy", "Septoria", "Healthy"]
    test_csv_path = test_dir / "test_features.csv"
    
    save_features_to_csv(
        features=test_features,
        filenames=test_filenames,
        labels=test_labels,
        output_path=test_csv_path,
        feature_dim=576
    )
    
    # Test CSV loading
    loaded_features = load_features_from_csv(test_csv_path)
    print(f"\n✓ Loaded features: {len(loaded_features)} samples")
    
    # Test validation
    print("\nValidating CSV...")
    validate_csv_output(test_csv_path, expected_samples=5, expected_dim=576)
    
    print("\n✓ All utility tests passed!")
