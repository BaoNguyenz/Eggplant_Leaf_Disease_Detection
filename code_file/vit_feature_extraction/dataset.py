"""
Dataset module for loading eggplant leaf disease images.
Handles image loading, preprocessing, and transformation.
"""
from pathlib import Path
from typing import Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import config


class EggplantDataset(Dataset):
    """
    Custom Dataset for loading eggplant leaf disease images.
    
    Directory structure expected:
    data_dir/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   └── ...
        └── ...
    """
    
    def __init__(
        self,
        data_dir: Path,
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to directory containing class folders
            image_size: Target image size (H, W) for resizing
            mean: ImageNet mean for normalization
            std: ImageNet std for normalization
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Collect all image paths and labels
        self.image_paths: List[Path] = []
        self.labels: List[str] = []
        
        self._load_image_paths()
        
        # Define transforms (ImageNet preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _load_image_paths(self) -> None:
        """
        Recursively load all image paths from class directories.
        Supports common image extensions: .jpg, .jpeg, .png, .bmp
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        
        # Iterate through class directories
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Find all images in this class directory
                for img_path in class_dir.rglob('*'):
                    if img_path.suffix in valid_extensions:
                        self.image_paths.append(img_path)
                        self.labels.append(class_name)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"[INFO] Found {len(self.image_paths)} images across {len(set(self.labels))} classes")
    
    def __len__(self) -> int:
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Get image, path, and label by index.
        
        Args:
            idx: Index of image to retrieve
            
        Returns:
            Tuple of (transformed_image, image_path_str, class_label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor, str(img_path), label


def create_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for feature extraction.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    dataset = EggplantDataset(
        data_dir=data_dir,
        image_size=config.IMAGE_SIZE,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for reproducibility
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
