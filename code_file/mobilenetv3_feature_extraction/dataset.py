"""
Dataset Module for MobileNetV3 Feature Extraction
=================================================
Custom PyTorch Dataset for loading and preprocessing images.
Handles Windows paths using pathlib and applies ImageNet normalization.

Author: AI Engineer
Date: 2026-02-04
"""

from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config


class EggplantDataset(Dataset):
    """
    Custom Dataset for loading eggplant leaf images.
    
    Attributes:
        image_paths (List[Path]): List of absolute paths to image files
        transform (transforms.Compose): Image preprocessing pipeline
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Initialize dataset by scanning directory for images.
        
        Args:
            data_dir (Path): Root directory containing images (can have subdirectories)
            transform (Optional[transforms.Compose]): Image transformations to apply
        
        Raises:
            FileNotFoundError: If data_dir does not exist
            ValueError: If no valid images are found
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self.data_dir = data_dir
        self.transform = transform or self._get_default_transform()
        
        # Supported image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Scan directory recursively for all images
        self.image_paths = self._collect_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No valid images found in {data_dir}\n"
                f"Supported formats: {self.valid_extensions}"
            )
        
        print(f"✓ Dataset initialized: {len(self.image_paths)} images found")
    
    def _collect_image_paths(self) -> List[Path]:
        """
        Recursively collect all image file paths from data directory.
        
        Returns:
            List[Path]: Sorted list of unique image paths
        
        Note:
            Uses set() to avoid duplicates on Windows (case-insensitive filesystem).
            For example, '*.jpg' and '*.JPG' would match the same file on Windows.
        """
        image_paths_set = set()  # Use set to ensure uniqueness
        
        # Use rglob for recursive search (handles subdirectories)
        # On Windows, file system is case-insensitive, so we only need lowercase
        for ext in self.valid_extensions:
            # Add all matches to set (automatically removes duplicates)
            image_paths_set.update(self.data_dir.rglob(f"*{ext}"))
            # Also search uppercase for cross-platform compatibility
            # (on Windows this will be duplicate, but set removes it)
            image_paths_set.update(self.data_dir.rglob(f"*{ext.upper()}"))
        
        # Convert set to sorted list for reproducibility
        return sorted(list(image_paths_set))
    
    def _get_default_transform(self) -> transforms.Compose:
        """
        Create default ImageNet preprocessing transform.
        
        Returns:
            transforms.Compose: Preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
        ])
    
    def __len__(self) -> int:
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Load and preprocess a single image.
        
        Args:
            idx (int): Index of image to load
        
        Returns:
            Tuple[torch.Tensor, str, str]: (preprocessed_image, image_filename, label)
                - preprocessed_image: Tensor of shape [3, 224, 224]
                - image_filename: Original filename (for CSV output)
                - label: Parent directory name (class label)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {e}")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)  # Shape: [3, 224, 224]
        
        # Get filename (without directory path)
        filename = img_path.name
        
        # Get label (parent directory name)
        label = img_path.parent.name
        
        return image, filename, label
    
    def get_relative_path(self, idx: int) -> str:
        """
        Get relative path from data_dir to image (useful for hierarchical datasets).
        
        Args:
            idx (int): Index of image
        
        Returns:
            str: Relative path string
        """
        img_path = self.image_paths[idx]
        try:
            relative = img_path.relative_to(self.data_dir)
            return str(relative)
        except ValueError:
            return img_path.name


def create_dataloader(
    data_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY
) -> DataLoader:
    """
    Create a DataLoader for feature extraction.
    
    Args:
        data_dir (Path): Directory containing images
        batch_size (int): Number of images per batch
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader: Configured DataLoader instance
    """
    dataset = EggplantDataset(data_dir=data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order for reproducibility
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Include all images, even if last batch is smaller
    )
    
    print(f"✓ DataLoader created: {len(dataset)} images, {len(dataloader)} batches")
    
    return dataloader


if __name__ == "__main__":
    # Test dataset module
    print("Testing Dataset Module...")
    print("-" * 70)
    
    # Create dataset
    dataset = EggplantDataset(data_dir=config.DATA_DIR)
    
    # Test single sample
    image, filename, label = dataset[0]
    print(f"\nSample Image:")
    print(f"  Shape    : {image.shape}")
    print(f"  Dtype    : {image.dtype}")
    print(f"  Range    : [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Filename : {filename}")
    print(f"  Label    : {label}")
    
    # Create dataloader
    print("\n" + "-" * 70)
    dataloader = create_dataloader(data_dir=config.DATA_DIR, batch_size=8)
    
    # Test batch loading
    batch_images, batch_filenames, batch_labels = next(iter(dataloader))
    print(f"\nSample Batch:")
    print(f"  Batch Shape : {batch_images.shape}")  # [B, 3, 224, 224]
    print(f"  Num Files   : {len(batch_filenames)}")
    print(f"  First File  : {batch_filenames[0]}")
    print(f"  First Label : {batch_labels[0]}")
