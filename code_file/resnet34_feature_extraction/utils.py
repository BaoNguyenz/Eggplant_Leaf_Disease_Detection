"""
Utility functions for feature extraction pipeline
"""
from pathlib import Path
from typing import List, Dict, Any
import csv
import numpy as np


def ensure_directory(path: Path) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of created/existing directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_directory(path: Path, dir_type: str = "directory") -> Path:
    """
    Validate that directory exists
    
    Args:
        path: Directory path to validate
        dir_type: Description of directory type (for error messages)
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If directory doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{dir_type} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{dir_type} is not a directory: {path}")
    return path


def validate_file(path: Path, file_type: str = "file") -> Path:
    """
    Validate that file exists
    
    Args:
        path: File path to validate
        file_type: Description of file type (for error messages)
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{file_type} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{file_type} is not a file: {path}")
    return path


def save_features_to_csv(
    features: List[np.ndarray],
    labels: List[str],
    filenames: List[str],
    output_path: Path,
    encoding: str = "utf-8"
) -> None:
    """
    Save extracted features to CSV file with UTF-8 encoding
    
    Args:
        features: List of feature vectors (numpy arrays)
        labels: List of class labels
        filenames: List of image filenames
        output_path: Output CSV file path
        encoding: File encoding (default: utf-8 for Windows compatibility)
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    ensure_directory(output_path.parent)
    
    # Determine feature dimension
    feature_dim = features[0].shape[0] if len(features) > 0 else 0
    
    # Create header
    header = ['filename', 'label'] + [f'feature_{i}' for i in range(feature_dim)]
    
    # Write CSV with UTF-8 encoding
    try:
        with open(output_path, 'w', newline='', encoding=encoding) as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write data rows
            for feat, label, filename in zip(features, labels, filenames):
                row = [filename, label] + feat.tolist()
                writer.writerow(row)
        
        print(f"✓ Features saved to: {output_path}")
        print(f"  └─ Total samples: {len(features)}")
        print(f"  └─ Feature dimension: {feature_dim}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV file: {e}")


def load_features_from_csv(csv_path: Path, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Load features from CSV file
    
    Args:
        csv_path: Path to CSV file
        encoding: File encoding
        
    Returns:
        Dictionary containing 'filenames', 'labels', and 'features'
    """
    csv_path = Path(csv_path)
    validate_file(csv_path, "CSV file")
    
    filenames = []
    labels = []
    features = []
    
    with open(csv_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            filenames.append(row[0])
            labels.append(row[1])
            features.append(np.array([float(x) for x in row[2:]]))
    
    return {
        'filenames': filenames,
        'labels': labels,
        'features': np.array(features)
    }


def print_extraction_summary(
    total_images: int,
    feature_dim: int,
    output_path: Path,
    elapsed_time: float
) -> None:
    """
    Print summary of feature extraction
    
    Args:
        total_images: Total number of images processed
        feature_dim: Dimension of feature vectors
        output_path: Path to output CSV file
        elapsed_time: Total elapsed time in seconds
    """
    print("\n" + "=" * 80)
    print("Feature Extraction Summary")
    print("=" * 80)
    print(f"Total images processed : {total_images}")
    print(f"Feature dimension      : {feature_dim}")
    print(f"Output file            : {output_path}")
    print(f"File size              : {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Elapsed time           : {elapsed_time:.2f} seconds")
    print(f"Processing speed       : {total_images/elapsed_time:.2f} images/sec")
    print("=" * 80)


if __name__ == "__main__":
    # Test utility functions
    import tempfile
    
    print("Testing utility functions...\n")
    
    # Test directory creation
    temp_dir = Path(tempfile.gettempdir()) / "test_resnet34_utils"
    ensure_directory(temp_dir)
    print(f"✓ Created directory: {temp_dir}")
    
    # Test CSV saving
    test_features = [np.random.randn(512) for _ in range(5)]
    test_labels = ['class_a', 'class_b', 'class_a', 'class_c', 'class_b']
    test_filenames = [f'image_{i}.jpg' for i in range(5)]
    
    csv_path = temp_dir / "test_features.csv"
    save_features_to_csv(test_features, test_labels, test_filenames, csv_path)
    
    # Test CSV loading
    loaded_data = load_features_from_csv(csv_path)
    print(f"\n✓ Loaded {len(loaded_data['features'])} feature vectors")
    print(f"✓ Feature shape: {loaded_data['features'].shape}")
    
    # Cleanup
    csv_path.unlink()
    temp_dir.rmdir()
    print(f"\n✓ Cleanup completed")
