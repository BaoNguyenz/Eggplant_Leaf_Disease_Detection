"""
Dataset module for ViT training pipeline.
Handles data loading, augmentation, class weight calculation, and stratified splitting.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def get_transforms(train: bool = True) -> v2.Compose:
    """
    Get image transformations for training or validation/test.
    
    Args:
        train: If True, return training transforms with augmentation.
               If False, return validation/test transforms without augmentation.
    
    Returns:
        Composed transformations
    """
    if train:
        # Strong augmentation for training
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=30),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.RandAugment(num_ops=2, magnitude=9),  # Strong augmentation
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Simple transforms for validation/test
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transforms


class SubsetDataset(Dataset):
    """
    Wrapper to create a subset dataset with custom transforms.
    """
    def __init__(self, dataset: Dataset, indices: List[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        image, label = self.dataset.imgs[original_idx]
        
        # Load image using PIL
        from PIL import Image
        image = Image.open(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def calculate_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights based on inverse frequency.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = np.bincount([label for _, label in dataset.imgs], minlength=num_classes)
    
    # Calculate inverse frequency weights
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize to prevent extreme values
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info("Class distribution:")
    for i, count in enumerate(class_counts):
        logger.info(f"  Class {i} ({dataset.classes[i]}): {count} samples, weight: {class_weights[i]:.4f}")
    
    return torch.FloatTensor(class_weights)


def create_weighted_sampler(dataset: Dataset, indices: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced training.
    
    Args:
        dataset: Original dataset
        indices: Indices of the subset to sample from
    
    Returns:
        WeightedRandomSampler for DataLoader
    """
    # Get labels for the subset
    labels = [dataset.imgs[idx][1] for idx in indices]
    
    # Calculate sample weights (inverse frequency)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
    """
    Create train, validation, and test dataloaders with stratified splitting.
    
    Args:
        args: Arguments from config.py
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights, class_names)
    """
    logger.info(f"Loading dataset from: {args.data_dir}")
    
    # Load full dataset without transforms (we'll apply them per split)
    full_dataset = datasets.ImageFolder(root=args.data_dir)
    
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    logger.info(f"Found {len(full_dataset)} images across {num_classes} classes")
    logger.info(f"Classes: {class_names}")
    
    # Get all labels for stratification
    all_labels = [label for _, label in full_dataset.imgs]
    all_indices = list(range(len(full_dataset)))
    
    # Calculate test split
    test_split = 1.0 - args.train_split - args.val_split
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_split,
        stratify=all_labels,
        random_state=args.seed
    )
    
    # Get labels for train+val subset
    train_val_labels = [all_labels[i] for i in train_val_indices]
    
    # Second split: separate train and val from train_val
    val_ratio = args.val_split / (args.train_split + args.val_split)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=args.seed
    )
    
    logger.info(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Calculate class weights from training set
    train_labels = [all_labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_indices)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights)
    
    logger.info("Training set class distribution and weights:")
    for i, count in enumerate(class_counts):
        logger.info(f"  {class_names[i]}: {count} samples, weight: {class_weights[i]:.4f}")
    
    # Create datasets with appropriate transforms
    train_dataset = SubsetDataset(full_dataset, train_indices, transform=get_transforms(train=True))
    val_dataset = SubsetDataset(full_dataset, val_indices, transform=get_transforms(train=False))
    test_dataset = SubsetDataset(full_dataset, test_indices, transform=get_transforms(train=False))
    
    # Create samplers
    train_sampler = None
    if args.use_weighted_sampler:
        logger.info("Using WeightedRandomSampler for training")
        train_sampler = create_weighted_sampler(full_dataset, train_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    return train_loader, val_loader, test_loader, class_weights, class_names


if __name__ == '__main__':
    # Test dataset creation
    from config import get_args
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    args = get_args()
    train_loader, val_loader, test_loader, class_weights, class_names = create_dataloaders(args)
    
    print(f"\nDataLoaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"\nClass weights: {class_weights}")
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
