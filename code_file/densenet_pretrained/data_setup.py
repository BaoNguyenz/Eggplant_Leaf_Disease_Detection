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


def create_dataloaders(data_dir, batch_size=32, num_workers=4, seed=42,
                       split_ratio=(0.8, 0.1, 0.1)):
    """
    Tạo DataLoader cho train/val/test với stratified split và data augmentation.
    Hỗ trợ tùy chỉnh tỷ lệ chia tập dữ liệu linh hoạt.
    
    Args:
        data_dir (str or Path): Đường dẫn đến thư mục chứa dữ liệu (Classified Images)
        batch_size (int): Kích thước batch
        num_workers (int): Số worker để load dữ liệu
        seed (int): Random seed cho reproducibility
        split_ratio (tuple): Tỷ lệ chia (train, val, test).
            - Ví dụ: (0.8, 0.1, 0.1) → 80% train, 10% val, 10% test
            - Ví dụ: (0.8, 0.2, 0.0) → 80% train, 20% val, KHÔNG có test
            - Tổng phải bằng 1.0 (hoặc script sẽ tự chuẩn hóa).
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, all_labels)
            - train_loader: DataLoader cho tập train
            - val_loader: DataLoader cho tập validation
            - test_loader: DataLoader cho tập test (None nếu test_ratio == 0)
            - class_names: List tên các lớp
            - all_labels: List tất cả nhãn (để tính class weights)
    """
    data_dir = Path(data_dir)
    
    # --- Chuẩn hóa split_ratio về tổng = 1.0 ---
    train_ratio, val_ratio, test_ratio = split_ratio
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError(f"Tổng split_ratio phải > 0, nhận được: {split_ratio}")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    print(f"\n[INFO] Split ratio (chuẩn hóa): "
          f"Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
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
    
    total_images = len(all_image_paths)
    print(f"\nTotal images: {total_images}")
    
    # Convert to numpy arrays
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    
    # ===================================================================
    # Stratified Split — Hỗ trợ 2 kịch bản: có/không tập Test
    # ===================================================================
    if test_ratio > 0:
        # --- Kịch bản 1: Có tập Test (chia 2 lần) ---
        # Bước 1: Tách Test ra khỏi toàn bộ dữ liệu
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=test_ratio,
            stratify=all_labels,
            random_state=seed
        )
        # Bước 2: Tách Val ra khỏi phần Train+Val còn lại
        # val_ratio_adjusted: tỷ lệ Val trong phần Train+Val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=seed
        )
    else:
        # --- Kịch bản 2: KHÔNG có tập Test (chỉ chia 1 lần) ---
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=val_ratio,
            stratify=all_labels,
            random_state=seed
        )
        test_paths = None
        test_labels = None
    
    # In thống kê chia tập
    print(f"\nDataset split (Train={train_ratio:.0%} / Val={val_ratio:.0%} / Test={test_ratio:.0%}):")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/total_images*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/total_images*100:.1f}%)")
    if test_paths is not None:
        print(f"  Test:  {len(test_paths)} images ({len(test_paths)/total_images*100:.1f}%)")
    else:
        print(f"  Test:  0 images (DISABLED — không chia tập test)")
    
    # Define transforms
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Image size
    image_size = 224
    
    # Train transform: Strong augmentation (enhanced version)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std_range=(0.0, 0.08)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
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
    
    # Test loader: chỉ tạo nếu có dữ liệu test
    if test_paths is not None:
        test_dataset = EggplantDataset(test_paths, test_labels, transform=val_test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader, class_names, all_labels.tolist()


if __name__ == "__main__":
    """
    Unit test cho data_setup.py — Kiểm tra 2 kịch bản chia tập:
      1. Có tập Test:  split_ratio = (0.8, 0.1, 0.1)
      2. Không Test:    split_ratio = (0.8, 0.2, 0.0)
    """
    data_dir = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images"
    
    # ============================================================
    # Test Case 1: split_ratio = (0.8, 0.1, 0.1) — CÓ tập test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST CASE 1: split_ratio = (0.8, 0.1, 0.1)")
    print("=" * 70)
    try:
        train_loader, val_loader, test_loader, class_names, all_labels = create_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=0,
            seed=42,
            split_ratio=(0.8, 0.1, 0.1)
        )
        
        print(f"\n✓ DataLoaders created successfully!")
        print(f"  Classes: {class_names}")
        print(f"  train_loader batches: {len(train_loader)}")
        print(f"  val_loader batches:   {len(val_loader)}")
        print(f"  test_loader:          {'None' if test_loader is None else f'{len(test_loader)} batches'}")
        
        # Kiểm tra test_loader phải tồn tại
        assert test_loader is not None, "FAIL: test_loader is None nhưng ratio > 0!"
        
        # Kiểm tra 1 batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print("✅ TEST CASE 1 PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST CASE 1 FAILED: {e}")
    
    # ============================================================
    # Test Case 2: split_ratio = (0.8, 0.2, 0.0) — KHÔNG có test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST CASE 2: split_ratio = (0.8, 0.2, 0.0)")
    print("=" * 70)
    try:
        train_loader2, val_loader2, test_loader2, class_names2, all_labels2 = create_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=0,
            seed=42,
            split_ratio=(0.8, 0.2, 0.0)
        )
        
        print(f"\n✓ DataLoaders created successfully!")
        print(f"  Classes: {class_names2}")
        print(f"  train_loader batches: {len(train_loader2)}")
        print(f"  val_loader batches:   {len(val_loader2)}")
        print(f"  test_loader:          {'None' if test_loader2 is None else f'{len(test_loader2)} batches'}")
        
        # Kiểm tra test_loader phải là None
        assert test_loader2 is None, "FAIL: test_loader không phải None nhưng ratio = 0!"
        
        # Kiểm tra 1 batch
        images2, labels2 = next(iter(train_loader2))
        print(f"  Batch shape: {images2.shape}, Labels shape: {labels2.shape}")
        print("✅ TEST CASE 2 PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST CASE 2 FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("All data_setup.py unit tests done.")
    print("=" * 70)
