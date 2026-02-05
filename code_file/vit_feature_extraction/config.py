"""
Configuration module for ViT Feature Extraction.
Contains all hyperparameters and path settings.
"""
from pathlib import Path
from typing import Tuple, List


class Config:
    """Configuration class for ViT feature extraction pipeline."""
    
    # ==================== PATH CONFIGURATION ====================
    # Dataset path (Classified Images)
    DATA_DIR: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images")
    
    # Output directory for extracted features
    OUTPUT_DIR: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction")
    
    # Output CSV filename
    OUTPUT_CSV: str = "vit_features.csv"
    
    # ==================== MODEL CONFIGURATION ====================
    # Model architecture: vit_b_16 (Vision Transformer Base, patch size 16)
    MODEL_NAME: str = "vit_b_16"
    
    # Pretrained weights: ImageNet-1K V1 or Custom trained weights
    PRETRAINED_WEIGHTS: str = "IMAGENET1K_V1"
    
    # Custom weights path (if you want to use your own trained model)
    # Set to None to use ImageNet pretrained weights
    CUSTOM_WEIGHTS_PATH: Path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vit_feature_extraction\vit_output\best_model.pth")
    
    # Use custom weights instead of ImageNet pretrained weights
    # Set to True to use CUSTOM_WEIGHTS_PATH, False to use ImageNet weights
    USE_CUSTOM_WEIGHTS: bool = True
    
    # Feature dimension (CLS token output from vit_b_16)
    FEATURE_DIM: int = 768
    
    # ==================== IMAGE PREPROCESSING ====================
    # Input image size for ViT (H, W)
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    
    # ImageNet normalization statistics
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
    
    # ==================== TRAINING CONFIGURATION ====================
    # Batch size for feature extraction
    BATCH_SIZE: int = 32
    
    # Number of workers for DataLoader (0 for Windows compatibility)
    NUM_WORKERS: int = 0
    
    # Pin memory for faster data transfer to GPU
    PIN_MEMORY: bool = True
    
    # ==================== DEVICE CONFIGURATION ====================
    # Auto-detect CUDA availability
    USE_CUDA: bool = True


# Create global config instance
config = Config()
