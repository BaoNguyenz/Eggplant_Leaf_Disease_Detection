"""
Dataset Module
Handles image loading and preprocessing for feature extraction.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import warnings


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from directory structure.
    
    Expected directory structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            ...
    """
    
    def __init__(self, data_dir: Path, transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Torchvision transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths, self.labels, self.class_names = self._load_image_paths()
        
        print(f"✓ Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
    
    def _load_image_paths(self) -> Tuple[List[Path], List[str], List[str]]:
        """
        Recursively load all image paths and their labels.
        
        Returns:
            tuple: (image_paths, labels, class_names)
        """
        image_paths = []
        labels = []
        class_names = []
        
        # Valid image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Iterate through subdirectories (each is a class)
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Find all images in this class directory
            for img_path in class_dir.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                    image_paths.append(img_path)
                    labels.append(class_name)
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        return image_paths, labels, sorted(set(class_names))
    
    def __len__(self) -> int:
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Path]:
        """
        Get image tensor, label, and path.
        
        Args:
            idx: Index of image
            
        Returns:
            tuple: (image_tensor, label, image_path)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
        
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {str(e)}")
            # Return a black image in case of error
            image = torch.zeros((3, 224, 224))
            return image, label, img_path


def custom_collate_fn(batch):
    """
    Custom collate function to handle Path objects in batch.
    
    This is necessary because PyTorch's default collate_fn cannot handle
    pathlib.Path objects when using multiple workers in DataLoader.
    
    Args:
        batch: List of tuples (image_tensor, label_str, image_path)
        
    Returns:
        Tuple of (batched_images, list_of_labels, list_of_paths_as_strings)
    """
    images = []
    labels = []
    paths = []
    
    for image, label, path in batch:
        images.append(image)
        labels.append(label)
        paths.append(str(path))  # Convert Path to string
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)
    
    return images, labels, paths


def get_transforms(img_size: Tuple[int, int], mean: List[float], std: List[float]) -> transforms.Compose:
    """
    Create image transformation pipeline.
    
    Args:
        img_size: Target image size (height, width)
        mean: Normalization mean values for RGB channels
        std: Normalization std values for RGB channels
        
    Returns:
        Composed transformations
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform


def create_dataloader(
    data_dir: Path,
    batch_size: int,
    img_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    num_workers: int = 4,
    shuffle: bool = False
) -> Tuple[DataLoader, ImageDataset]:
    """
    Create DataLoader for batch processing.
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size for DataLoader
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        tuple: (DataLoader, Dataset)
    """
    # Create transforms
    transform = get_transforms(img_size, mean, std)
    
    # Create dataset
    dataset = ImageDataset(data_dir, transform=transform)
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn  # Use custom collate to handle Path objects
    )
    
    return dataloader, dataset
