"""
Dataset module for loading and preprocessing images
"""
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from directory structure
    
    Expected structure:
        data_dir/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image3.jpg
    """
    
    def __init__(
        self, 
        data_dir: Path, 
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing class folders
            transform: Optional torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: List[Path] = []
        self.labels: List[str] = []
        
        # Validate directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Load all image paths
        self._load_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def _load_image_paths(self) -> None:
        """Scan directory and collect all image paths"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Iterate through class directories
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Collect all images in this class
            for img_path in class_dir.rglob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(img_path)
                    self.labels.append(class_name)
    
    def __len__(self) -> int:
        """Return total number of images"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Get image by index
        
        Args:
            idx: Index of image
            
        Returns:
            Tuple of (transformed_image, label, image_filename)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image tensor, label, and filename
        return image, label, img_path.name
    
    def get_class_distribution(self) -> dict:
        """Return distribution of classes in dataset"""
        from collections import Counter
        return dict(Counter(self.labels))


def get_transform(image_size: Tuple[int, int], mean: Tuple[float, ...], std: Tuple[float, ...]) -> transforms.Compose:
    """
    Create preprocessing transform pipeline
    
    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


if __name__ == "__main__":
    # Test dataset loading
    from config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
    
    transform = get_transform(IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD)
    dataset = ImageDataset(DATA_DIR, transform=transform)
    
    print(f"Total images: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test loading first image
    img, label, filename = dataset[0]
    print(f"\nSample: {filename}")
    print(f"Label: {label}")
    print(f"Tensor shape: {img.shape}")
