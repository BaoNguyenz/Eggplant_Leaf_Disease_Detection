"""
Configuration Module for EfficientNet-B0 Feature Extraction
============================================================
Quản lý tất cả đường dẫn, tham số và cấu hình hệ thống.
Tối ưu cho Windows với pathlib.
"""

from pathlib import Path
from typing import Tuple
import torch


class Config:
    """
    Centralized configuration class cho feature extraction pipeline.
    
    Attributes:
        DATA_DIR: Đường dẫn đến thư mục chứa dataset (Classified Images)
        OUTPUT_DIR: Đường dẫn lưu file CSV kết quả
        BATCH_SIZE: Kích thước batch (thấp do model nặng)
        NUM_WORKERS: Số worker cho DataLoader
        IMG_SIZE: Kích thước ảnh input (224x224 tối ưu cho B0)
        DEVICE: CUDA hoặc CPU
        IMAGENET_MEAN: Mean normalization ImageNet
        IMAGENET_STD: Std normalization ImageNet
        FEATURE_DIM: Số chiều vector đặc trưng (1280 cho EfficientNet-B0)
    """
    
    # ==================== DEFAULT PATHS ====================
    DATA_DIR: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images")
    OUTPUT_DIR: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction")
    CUSTOM_WEIGHTS_PATH: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction\effcientnetb0_output\best_model.pth")  # Custom trained B0 weights
    
    # ==================== MODEL PARAMETERS ====================
    BATCH_SIZE: int = 8  # Batch size cao hơn do EfficientNet-B0 nhẹ (~20M params)
    NUM_WORKERS: int = 4  # Số worker cho DataLoader (tuỳ chỉnh theo CPU cores)
    IMG_SIZE: Tuple[int, int] = (224, 224)  # Resolution tối ưu cho EfficientNet-B0
    FEATURE_DIM: int = 1280  # Output dimension của EfficientNet-B0 features
    
    # ==================== DEVICE CONFIGURATION ====================
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==================== IMAGENET NORMALIZATION ====================
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    @classmethod
    def update_paths(cls, data_dir: str = None, output_dir: str = None) -> None:
        """
        Cập nhật đường dẫn từ command line arguments.
        
        Args:
            data_dir: Đường dẫn tới thư mục dataset
            output_dir: Đường dẫn lưu output
        """
        if data_dir:
            cls.DATA_DIR = Path(data_dir)
        if output_dir:
            cls.OUTPUT_DIR = Path(output_dir)
    
    @classmethod
    def update_batch_size(cls, batch_size: int) -> None:
        """
        Cập nhật batch size từ command line.
        
        Args:
            batch_size: Kích thước batch mới
        """
        cls.BATCH_SIZE = batch_size
    
    @classmethod
    def update_custom_weights(cls, custom_weights: str) -> None:
        """
        Cập nhật đường dẫn custom weights từ command line.
        
        Args:
            custom_weights: Đường dẫn tới file .pth checkpoint
        """
        if custom_weights:
            cls.CUSTOM_WEIGHTS_PATH = Path(custom_weights)
    
    @classmethod
    def validate_paths(cls) -> None:
        """
        Kiểm tra tính hợp lệ của đường dẫn input.
        
        Raises:
            FileNotFoundError: Nếu DATA_DIR không tồn tại
        """
        if not cls.DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {cls.DATA_DIR}")
        
        if not cls.DATA_DIR.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {cls.DATA_DIR}")
    
    @classmethod
    def get_info(cls) -> str:
        """
        Trả về thông tin cấu hình dưới dạng string.
        
        Returns:
            String chứa thông tin cấu hình
        """
        weights_info = f"Custom: {cls.CUSTOM_WEIGHTS_PATH}" if cls.CUSTOM_WEIGHTS_PATH else "ImageNet Pretrained"
        return f"""
========================================
EfficientNet-B0 Feature Extraction Config
========================================
Data Directory    : {cls.DATA_DIR}
Output Directory  : {cls.OUTPUT_DIR}
Batch Size        : {cls.BATCH_SIZE}
Num Workers       : {cls.NUM_WORKERS}
Image Size        : {cls.IMG_SIZE}
Feature Dim       : {cls.FEATURE_DIM}
Device            : {cls.DEVICE}
Weights Source    : {weights_info}
ImageNet Mean     : {cls.IMAGENET_MEAN}
ImageNet Std      : {cls.IMAGENET_STD}
========================================
        """
