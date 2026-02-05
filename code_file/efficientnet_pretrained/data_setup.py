"""
Data pipeline for EfficientNet-B0 (224x224 input resolution).
Includes custom augmentation with AddGaussianNoise and stratified split.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


class AddGaussianNoise(object):
    """
    Custom transform to add Gaussian noise with random standard deviation.
    """
    
    def __init__(self, mean: float = 0.0, std_range: tuple = (0, 0.05)):
        """
        Args:
            mean: Mean of Gaussian noise
            std_range: Range for random std (min, max). Max 0.05 = 12.75/255
        """
        self.mean = mean
        self.std_min, self.std_max = std_range
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Input tensor [C, H, W] (normalized)
            
        Returns:
            Tensor with added Gaussian noise
        """
        # Random std between 0 and 0.05
        std = np.random.uniform(self.std_min, self.std_max)
        noise = torch.randn_like(tensor) * std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std_range=({self.std_min}, {self.std_max}))"


class EggplantDataset(Dataset):
    """
    Custom Dataset for Eggplant Leaf Disease Classification.
    """
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of integer labels
            transform: Torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.targets = labels  # For compatibility with get_class_weights
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(data_dir: str, 
                      batch_size: int = 8, 
                      num_workers: int = 4,
                      seed: int = 42) -> tuple:
    """
    Create train, validation, and test dataloaders with stratified split.
    
    Args:
        data_dir: Root directory containing class folders
        batch_size: Batch size (default: 8 for 600x600 images)
        num_workers: Number of workers for DataLoader
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, num_classes)
    """
    data_dir = Path(data_dir)
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Data Directory: {data_dir}")
    print(f"Number of Classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        class_idx = class_to_idx[class_name]
        
        # Collect images (jpg, jpeg, png)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in class_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(class_idx)
    
    print(f"Total Images: {len(image_paths)}")
    print("="*60 + "\n")
    
    # Stratified split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    print("Split Statistics:")
    print(f"  Train: {len(X_train)} images")
    print(f"  Val:   {len(X_val)} images")
    print(f"  Test:  {len(X_test)} images\n")
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # =====================================
    # TRAIN TRANSFORMS (224x224 with Augmentation)
    # =====================================
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.5),               # Horizontal flip
        transforms.RandomRotation(degrees=15),                # Rotation ±15°
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0)),  # Blur
        transforms.ColorJitter(brightness=(0.8, 1.2)),        # Brightness 80%-120%
        transforms.ToTensor(),                                # Convert to tensor [0, 1]
        AddGaussianNoise(mean=0, std_range=(0, 0.05)),       # Custom Gaussian noise
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # =====================================
    # VAL/TEST TRANSFORMS (224x224, no augmentation)
    # =====================================
    eval_transforms = transforms.Compose([
        transforms.Resize(224),                              # Resize shorter side to 224
        transforms.CenterCrop(224),                          # Crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Create datasets
    train_dataset = EggplantDataset(X_train, y_train, transform=train_transforms)
    val_dataset = EggplantDataset(X_val, y_val, transform=eval_transforms)
    test_dataset = EggplantDataset(X_test, y_test, transform=eval_transforms)
    
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
    
    print("✓ Dataloaders created successfully\n")
    
    return train_loader, val_loader, test_loader, class_names, len(class_names)
