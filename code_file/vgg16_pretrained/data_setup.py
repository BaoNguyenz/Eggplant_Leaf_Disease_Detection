"""
Data setup module for VGG16 Eggplant Leaf Disease Detection.
Contains dataset class and dataloader creation with stratified splitting.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EggplantDataset(Dataset):
    """
    Custom Dataset for Eggplant Leaf Disease Detection.
    
    Args:
        image_paths: List of paths to images.
        labels: List of corresponding labels.
        transform: Optional transform to apply to images.
        class_to_idx: Dictionary mapping class names to indices.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (image_tensor, label).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.
    
    Args:
        image_size: Target image size.
        is_training: Whether to apply training augmentations.
        
    Returns:
        Composed transforms.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def load_dataset_from_folder(data_dir: str) -> Tuple[List[str], List[int], Dict[str, int], List[str]]:
    """
    Load image paths and labels from a folder structure.
    Expects structure: data_dir/class_name/image.jpg
    
    Args:
        data_dir: Root directory of the dataset.
        
    Returns:
        Tuple of (image_paths, labels, class_to_idx, class_names).
    """
    data_path = Path(data_dir)
    
    # Get class names from folder names
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    for class_name in class_names:
        class_dir = data_path / class_name
        class_idx = class_to_idx[class_name]
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(class_idx)
    
    print(f"[INFO] Found {len(image_paths)} images in {len(class_names)} classes")
    print(f"[INFO] Classes: {class_names}")
    
    return image_paths, labels, class_to_idx, class_names


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], List[str], EggplantDataset]:
    """
    Create train, validation, and test dataloaders with stratified splitting.
    
    Args:
        data_dir: Root directory of the dataset.
        batch_size: Batch size for dataloaders.
        image_size: Target image size.
        num_workers: Number of worker processes for data loading.
        train_ratio: Ratio of training data (default: 0.8).
        val_ratio: Ratio of validation data (default: 0.1).
        test_ratio: Ratio of test data (default: 0.1).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx, class_names, train_dataset).
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load all data
    image_paths, labels, class_to_idx, class_names = load_dataset_from_folder(data_dir)
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        image_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    test_size_relative = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels,
        test_size=test_size_relative,
        stratify=val_test_labels,
        random_state=random_state
    )
    
    print(f"[INFO] Dataset split (Stratified 80/10/10):")
    print(f"       Train: {len(train_paths)} samples")
    print(f"       Val:   {len(val_paths)} samples")
    print(f"       Test:  {len(test_paths)} samples")
    
    # Create transforms
    train_transform = get_transforms(image_size=image_size, is_training=True)
    eval_transform = get_transforms(image_size=image_size, is_training=False)
    
    # Create datasets
    train_dataset = EggplantDataset(
        train_paths, train_labels,
        transform=train_transform,
        class_to_idx=class_to_idx
    )
    val_dataset = EggplantDataset(
        val_paths, val_labels,
        transform=eval_transform,
        class_to_idx=class_to_idx
    )
    test_dataset = EggplantDataset(
        test_paths, test_labels,
        transform=eval_transform,
        class_to_idx=class_to_idx
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_to_idx, class_names, train_dataset
