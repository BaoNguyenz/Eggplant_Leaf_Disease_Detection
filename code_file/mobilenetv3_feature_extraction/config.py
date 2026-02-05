"""
Configuration Module for MobileNetV3 Feature Extraction
========================================================
Centralized configuration for paths, model parameters, and device settings.
Optimized for Windows with pathlib to handle long paths and backslashes.

Author: AI Engineer
Date: 2026-02-04
"""

from pathlib import Path
from typing import Tuple
import torch


# ============================================================================
# DIRECTORY PATHS (Using pathlib for Windows compatibility)
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset")

# Dataset directory (Classified Images)
DATA_DIR = PROJECT_ROOT / "Eggplant Dataset" / "Classified Images"

# Output directory for extracted features
OUTPUT_DIR = PROJECT_ROOT / "code_file" / "mobilenetv3_feature_extraction" / "extracted_features"

# Model weights directory
MODEL_DIR = PROJECT_ROOT / "code_file" / "mobilenetv3_feature_extraction" / "mobilenetv3_output"

# Custom trained weights path
MODEL_WEIGHT_PATH = MODEL_DIR / "best_model.pth"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# MobileNetV3 Small architecture parameters
MODEL_NAME: str = "mobilenet_v3_small"
FEATURE_DIM: int = 576  # Output dimension after avgpool (MobileNetV3 Small standard)

# Image preprocessing parameters (ImageNet standard)
IMG_SIZE: Tuple[int, int] = (224, 224)
IMG_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMG_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================

BATCH_SIZE: int = 32
NUM_WORKERS: int = 4  # Adjust based on CPU cores (0 for debugging on Windows)
PIN_MEMORY: bool = True  # Set to True if using GPU


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: CUDA GPU if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    return device


DEVICE = get_device()


# ============================================================================
# FILE NAMING CONFIGURATION
# ============================================================================

OUTPUT_CSV_NAME: str = "mobilenetv3_features.csv"
OUTPUT_CSV_PATH = OUTPUT_DIR / OUTPUT_CSV_NAME


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_paths() -> None:
    """
    Validate that all critical paths exist.
    Raises FileNotFoundError if any required path is missing.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    
    if not MODEL_WEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"Model weights file not found: {MODEL_WEIGHT_PATH}\n"
            f"Please ensure you have trained the model and saved weights."
        )
    
    print(f"✓ Dataset directory validated: {DATA_DIR}")
    print(f"✓ Model weights validated: {MODEL_WEIGHT_PATH}")


def print_config() -> None:
    """Print current configuration for verification."""
    print("\n" + "="*70)
    print("MOBILENETV3 FEATURE EXTRACTION CONFIGURATION")
    print("="*70)
    print(f"Model Architecture    : {MODEL_NAME}")
    print(f"Feature Dimension     : {FEATURE_DIM}")
    print(f"Image Size            : {IMG_SIZE}")
    print(f"Batch Size            : {BATCH_SIZE}")
    print(f"Num Workers           : {NUM_WORKERS}")
    print(f"Device                : {DEVICE}")
    print(f"Data Directory        : {DATA_DIR}")
    print(f"Output Directory      : {OUTPUT_DIR}")
    print(f"Model Weights         : {MODEL_WEIGHT_PATH}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config()
    validate_paths()
