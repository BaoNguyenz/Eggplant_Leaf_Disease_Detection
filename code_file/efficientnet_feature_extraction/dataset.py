"""
Dataset Module for EfficientNet-B0 Feature Extraction
=====================================================
Custom Dataset class với transforms tối ưu cho EfficientNet-B0.
"""

from pathlib import Path
from typing import List, Tuple, Callable, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class EggplantLeafDataset(Dataset):
    """
    Custom Dataset cho ảnh lá cà tím.
    Load ảnh và áp dụng transforms cho EfficientNet-B7.
    
    Attributes:
        image_paths: List các Path tới file ảnh
        transform: Transforms áp dụng lên ảnh
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        transform: Optional[Callable] = None
    ) -> None:
        """
        Khởi tạo Dataset.
        
        Args:
            image_paths: List đường dẫn tới ảnh
            transform: Torchvision transforms (optional)
        """
        self.image_paths = image_paths
        self.transform = transform
        
        if len(self.image_paths) == 0:
            raise ValueError("No images found in the dataset!")
    
    def __len__(self) -> int:
        """Trả về số lượng ảnh trong dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load và transform ảnh tại index.
        
        Args:
            idx: Index của ảnh cần load
        
        Returns:
            Tuple (transformed_image, image_path_string)
        
        Note:
            image_path được convert thành string để tương thích với
            PyTorch DataLoader collate function (không hỗ trợ Path objects)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Cannot load image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert Path to string để tránh lỗi collate trong DataLoader
        return image, str(img_path)


def get_efficientnet_b0_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Tạo transforms pipeline cho EfficientNet-B0.
    
    Pipeline:
    1. Resize to (224, 224) - Resolution tối ưu cho B0
    2. Convert to Tensor
    3. Normalize với ImageNet mean/std
    
    Args:
        img_size: Kích thước ảnh đầu ra
        mean: Mean cho normalization (ImageNet)
        std: Std cho normalization (ImageNet)
    
    Returns:
        Composed transforms
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize về 224x224
        transforms.ToTensor(),  # Convert PIL Image to Tensor [C, H, W] (0-1)
        transforms.Normalize(mean=mean, std=std)  # Normalize với ImageNet statistics
    ])
    
    return transform


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """
    Tạo DataLoader từ Dataset.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Kích thước batch (cao hơn cho EfficientNet-B0)
        num_workers: Số worker processes
        shuffle: Có shuffle data không
    
    Returns:
        DataLoader instance
    """
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return dataloader
