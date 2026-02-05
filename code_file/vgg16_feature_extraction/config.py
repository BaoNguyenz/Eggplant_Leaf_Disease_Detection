"""
Configuration Module for VGG16 Feature Extraction
Contains all hyperparameters and path configurations.
"""
from pathlib import Path


class Config:
    """Configuration class for VGG16 feature extraction pipeline."""
    
    # Image preprocessing parameters
    IMG_SIZE = (224, 224)  # VGG16 input size
    MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization mean
    STD = [0.229, 0.224, 0.225]   # ImageNet normalization std
    
    # Model parameters
    MODEL_NAME = "VGG16"
    USE_PRETRAINED = True
    FEATURE_DIM = 512  # Output dimension after Global Average Pooling
    
    # Custom weights path - LOAD YOUR FINE-TUNED CHECKPOINT
    CUSTOM_WEIGHTS_PATH = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vgg16_pretrained\vgg16_results\vgg16_best.pth")
    
    # Data processing
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # For DataLoader
    
    # File formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    OUTPUT_ENCODING = 'utf-8'
    
    # Default paths (can be overridden via CLI)
    DEFAULT_DATA_DIR = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images")
    DEFAULT_OUTPUT_DIR = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\vgg16_feature_extraction")
    
    @staticmethod
    def validate_paths(data_dir: Path, output_dir: Path) -> tuple:
        """
        Validate input and output directories.
        
        Args:
            data_dir: Path to dataset directory
            output_dir: Path to output directory
            
        Returns:
            tuple: (data_dir, output_dir) as Path objects
            
        Raises:
            FileNotFoundError: If data_dir does not exist
        """
        data_dir = Path(data_dir).resolve()
        output_dir = Path(output_dir).resolve()
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return data_dir, output_dir
