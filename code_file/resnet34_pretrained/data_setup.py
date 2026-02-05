"""
Data pipeline for eggplant leaf disease detection.
Handles dataset loading, custom transforms, and dataloader creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


class AddGaussianNoise:
    """
    Custom transform to add Gaussian noise to images.
    """
    
    def __init__(self, mean=0.0, std=0.1):
        """
        Args:
            mean (float): Mean of Gaussian noise
            std (float): Standard deviation of Gaussian noise
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Image tensor to add noise to
            
        Returns:
            Tensor: Image with added Gaussian noise
        """
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class EggplantDataset(Dataset):
    """
    Custom Dataset for eggplant leaf images organized in class folders.
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str or Path): Root directory containing class subdirectories
            transform (callable, optional): Transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples = []
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, class_idx))
        
        self.targets = [s[1] for s in self.samples]
        print(f"✓ Loaded {len(self.samples)} images from {len(self.classes)} classes")
        print(f"  Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(data_dir, batch_size=32, num_workers=2, seed=42):
    """
    Create train, validation, and test dataloaders with appropriate transforms.
    
    Args:
        data_dir (str or Path): Root directory containing class subdirectories
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker threads for data loading
        seed (int): Random seed for reproducible splits
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, full_dataset)
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1),
        normalize
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load full dataset
    full_dataset = EggplantDataset(data_dir, transform=None)
    class_names = full_dataset.classes
    
    # Calculate split sizes (80% train, 10% val, 10% test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Apply transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    print(f"\n✓ Dataset split (seed={seed}):")
    print(f"  Train: {train_size} samples ({train_size/total_size*100:.1f}%)")
    print(f"  Val:   {val_size} samples ({val_size/total_size*100:.1f}%)")
    print(f"  Test:  {test_size} samples ({test_size/total_size*100:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    return train_loader, val_loader, test_loader, class_names, full_dataset
