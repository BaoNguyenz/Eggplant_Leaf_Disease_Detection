"""
Configuration file for ResNet34 Feature Extraction
Manages paths, hyperparameters, and device settings
"""
from pathlib import Path
import torch
from typing import Tuple


# ============================================================================
# PATH CONFIGURATION (Using pathlib for Windows compatibility)
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Dataset directory
DATA_DIR = PROJECT_ROOT / "Eggplant Dataset" / "Classified Images"

# Output directory for features
OUTPUT_DIR = BASE_DIR / "extracted_features"

# Custom model weights path
MODEL_WEIGHT_PATH = (
    PROJECT_ROOT / "code_file" / "resnet34_feature_extract" / 
    "Resnet34_output" / "best_model.pth"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Image preprocessing parameters (ImageNet standards)
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Feature dimension for ResNet34 (after removing FC layer)
FEATURE_DIM: int = 512


# ============================================================================
# TRAINING/INFERENCE PARAMETERS
# ============================================================================

# Batch size for feature extraction
BATCH_SIZE: int = 32

# Number of workers for DataLoader (set to 0 on Windows if issues occur)
NUM_WORKERS: int = 4

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output CSV filename
OUTPUT_CSV_NAME: str = "resnet34_features.csv"

# CSV encoding (UTF-8 for Windows compatibility)
CSV_ENCODING: str = "utf-8"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_config() -> None:
    """Print current configuration settings"""
    print("=" * 80)
    print("ResNet34 Feature Extraction - Configuration")
    print("=" * 80)
    print(f"Data Directory      : {DATA_DIR}")
    print(f"Output Directory    : {OUTPUT_DIR}")
    print(f"Model Weights Path  : {MODEL_WEIGHT_PATH}")
    print(f"Image Size          : {IMAGE_SIZE}")
    print(f"Feature Dimension   : {FEATURE_DIM}")
    print(f"Batch Size          : {BATCH_SIZE}")
    print(f"Number of Workers   : {NUM_WORKERS}")
    print(f"Device              : {DEVICE}")
    print(f"Output CSV          : {OUTPUT_CSV_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
