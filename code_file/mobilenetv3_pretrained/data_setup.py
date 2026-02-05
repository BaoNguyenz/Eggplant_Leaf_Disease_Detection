"""
Data Setup Module for MobileNetV3 Training
Handles dataset loading, augmentation, and dataloader creation
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


class AddGaussianNoise(object):
    """
    Add Gaussian noise to image tensor
    
    Args:
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
    """
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Image tensor to add noise to
            
        Returns:
            Tensor: Noisy image tensor
        """
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def create_dataloaders(data_dir, batch_size=32, num_workers=2, seed=42):
    """
    Create train, validation, and test dataloaders with appropriate transforms
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducible splitting
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset, class_names)
    """
    
    # ImageNet normalization parameters
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.05),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Load full dataset with training transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"[INFO] Found {num_classes} classes: {class_names}")
    print(f"[INFO] Total images: {len(full_dataset)}")
    
    # Calculate split sizes (80% train, 10% val, 10% test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"[INFO] Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Set seed for reproducible split
    torch.manual_seed(seed)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply appropriate transforms to validation and test sets
    # Note: We need to create new datasets with different transforms
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)
    test_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"[INFO] Dataloaders created successfully")
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, full_dataset, class_names
