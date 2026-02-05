"""
Utility Module for EfficientNet-B7 Feature Extraction
=====================================================
Các hàm phụ trợ: kiểm tra thư mục, lưu CSV, xử lý path Windows.
"""

from pathlib import Path
from typing import List, Dict
import csv
import torch
import numpy as np


def ensure_dir(directory: Path) -> None:
    """
    Tạo thư mục nếu chưa tồn tại (Windows compatible).
    
    Args:
        directory: Đường dẫn thư mục cần tạo
    """
    directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory ensured: {directory}")


def get_image_paths(data_dir: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """
    Lấy danh sách đường dẫn tất cả ảnh trong thư mục dataset.
    Hỗ trợ cấu trúc thư mục phân cấp (class/image.jpg).
    
    Args:
        data_dir: Thư mục gốc chứa dataset
        extensions: Tuple các extension ảnh hợp lệ
    
    Returns:
        List các Path đến file ảnh
    """
    image_paths = []
    
    # Duyệt qua tất cả subfolder (class folders)
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            # Lấy tất cả ảnh trong class folder
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in extensions:
                    image_paths.append(img_path)
    
    print(f"✓ Found {len(image_paths)} images in {data_dir}")
    return image_paths


def extract_class_from_path(image_path: Path, data_dir: Path) -> str:
    """
    Trích xuất tên class từ đường dẫn ảnh (parent folder name).
    
    Args:
        image_path: Đường dẫn tới ảnh
        data_dir: Thư mục gốc dataset
    
    Returns:
        Tên class (parent folder name)
    """
    # Lấy relative path và extract parent folder name
    relative_path = image_path.relative_to(data_dir)
    class_name = relative_path.parent.name
    return class_name


def save_features_to_csv(
    features: torch.Tensor,
    image_paths: List,  # Có thể là List[Path] hoặc List[str]
    data_dir: Path,
    output_path: Path,
    feature_dim: int = 2560
) -> None:
    """
    Lưu features vào file CSV với encoding UTF-8 (Windows compatible).
    
    Format CSV:
    - Row 1: Header (image_path, class_name, feature_1, feature_2, ..., feature_2560)
    - Row 2+: Data (relative_path, class, f1, f2, ..., f2560)
    
    Args:
        features: Tensor chứa features [N, feature_dim]
        image_paths: List đường dẫn ảnh (Path objects hoặc strings)
        data_dir: Thư mục gốc dataset
        output_path: Đường dẫn file CSV output
        feature_dim: Số chiều feature vector (2560 cho EfficientNet-B7)
    """
    # Convert tensor to numpy
    features_np = features.cpu().numpy()
    
    # Mở file CSV với encoding utf-8
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['image_path', 'class_name'] + [f'feature_{i+1}' for i in range(feature_dim)]
        writer.writerow(header)
        
        # Write data rows
        for img_path, feature_vector in zip(image_paths, features_np):
            # Convert string to Path if needed (từ DataLoader trả về strings)
            if isinstance(img_path, str):
                img_path = Path(img_path)
            
            # Lấy relative path (tương đối so với data_dir)
            relative_path = img_path.relative_to(data_dir)
            # Convert WindowsPath to string với forward slashes
            relative_path_str = str(relative_path).replace('\\', '/')
            
            # Extract class name
            class_name = extract_class_from_path(img_path, data_dir)
            
            # Tạo row: [path, class, f1, f2, ..., f2560]
            row = [relative_path_str, class_name] + feature_vector.tolist()
            writer.writerow(row)
    
    print(f"✓ Features saved to CSV: {output_path}")


def validate_features(features: torch.Tensor, expected_dim: int = 2560) -> None:
    """
    Kiểm tra tính hợp lệ của feature tensor.
    
    Args:
        features: Tensor features cần validate
        expected_dim: Số chiều mong đợi (2560 cho EfficientNet-B7)
    
    Raises:
        ValueError: Nếu feature dimension không đúng
    """
    if features.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {features.dim()}D")
    
    if features.shape[1] != expected_dim:
        raise ValueError(f"Expected feature dim {expected_dim}, got {features.shape[1]}")
    
    # Check for NaN or Inf
    if torch.isnan(features).any():
        raise ValueError("Features contain NaN values")
    
    if torch.isinf(features).any():
        raise ValueError("Features contain Inf values")
    
    print(f"✓ Features validation passed: {features.shape}")


def get_device_info() -> Dict[str, str]:
    """
    Lấy thông tin về device đang sử dụng.
    
    Returns:
        Dictionary chứa thông tin device
    """
    device_info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": str(torch.cuda.is_available()),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        })
    
    return device_info


def print_device_info() -> None:
    """In thông tin device ra console."""
    info = get_device_info()
    print("\n========================================")
    print("Device Information")
    print("========================================")
    for key, value in info.items():
        print(f"{key:15}: {value}")
    print("========================================\n")
