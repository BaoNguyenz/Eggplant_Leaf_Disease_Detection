"""
Data Setup for DenseNet121 Training Pipeline
Includes: EggplantDataset class and DataLoader creation with stratified splitting
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


class EggplantDataset(Dataset):
    """
    Custom Dataset cho ảnh lá cà tím (Eggplant Leaf Disease Detection).
    
    Args:
        image_paths (list): Danh sách đường dẫn đến các ảnh
        labels (list): Danh sách nhãn tương ứng
        transform (torchvision.transforms): Các phép biến đổi ảnh
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AddGaussianNoise:
    """
    Custom transform để thêm Gaussian Noise vào ảnh với cường độ ngẫu nhiên.
    Theo bảng augmentation: scale=(0, 0.05 * 255) → std random trong [0, 0.05] sau khi ToTensor().
    
    Args:
        mean (float): Giá trị trung bình của noise
        std_range (tuple): Range của std để random (min_std, max_std)
    """
    def __init__(self, mean=0.0, std_range=(0.0, 0.05)):
        self.mean = mean
        self.std_range = std_range
    
    def __call__(self, tensor):
        # Random std từ range mỗi lần gọi
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range})'


def create_dataloaders(data_dir, batch_size=32, num_workers=4, seed=42):
    """
    Tạo DataLoader cho train/val/test với stratified split và data augmentation.
    
    Args:
        data_dir (str or Path): Đường dẫn đến thư mục chứa dữ liệu (Classified Images)
        batch_size (int): Kích thước batch
        num_workers (int): Số worker để load dữ liệu
        seed (int): Random seed cho reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, all_labels)
            - train_loader: DataLoader cho tập train
            - val_loader: DataLoader cho tập validation
            - test_loader: DataLoader cho tập test
            - class_names: List tên các lớp
            - all_labels: List tất cả nhãn (để tính class weights)
    """
    data_dir = Path(data_dir)
    
    # Lấy danh sách các lớp (thư mục con)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Thu thập tất cả đường dẫn ảnh và nhãn
    all_image_paths = []
    all_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        # Tìm tất cả ảnh trong thư mục (jpg, jpeg, png)
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))
        
        print(f"Class '{class_names[class_idx]}': {len(image_files)} images")
        
        for img_path in image_files:
            all_image_paths.append(img_path)
            all_labels.append(class_idx)
    
    print(f"\nTotal images: {len(all_image_paths)}")
    
    # Convert to numpy arrays
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    
    
    # Stratified split: 75% train, 15% val, 10% test
    # First split: 75% train, 25% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=0.25,
        stratify=all_labels,
        random_state=seed
    )
    
    # Second split: 60% val, 40% test (from temp 25% → 15% val, 10% test)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.4,
        stratify=temp_labels,
        random_state=seed
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(all_image_paths)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/len(all_image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/len(all_image_paths)*100:.1f}%)")
    
    # Define transforms
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Image size
    image_size = 224
    
    # Train transform: Strong augmentation (enhanced version)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # NEW: Vertical flip
        transforms.RandomRotation(degrees=25),  # Increased from 20 to 25
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.15),  # More variation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW: Random translation
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Stronger blur
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std_range=(0.0, 0.08)),  # Increased noise
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # NEW: Random erasing
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Val/Test transform: Minimal augmentation (only resize and normalize)
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Create datasets
    train_dataset = EggplantDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = EggplantDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = EggplantDataset(test_paths, test_labels, transform=val_test_transform)
    
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
    
    return train_loader, val_loader, test_loader, class_names, all_labels.tolist()


if __name__ == "__main__":
    # Test data loading
    print("Testing data_setup.py...")
    
    # Example data directory (adjust path)
    data_dir = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images"
    
    try:
        train_loader, val_loader, test_loader, class_names, all_labels = create_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=0,  # Use 0 for testing on Windows
            seed=42
        )
        
        print(f"\n✓ DataLoaders created successfully!")
        print(f"Classes: {class_names}")
        
        # Test loading one batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
