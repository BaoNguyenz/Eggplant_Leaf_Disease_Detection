"""
Model setup module for VGG16 Eggplant Leaf Disease Detection.
Contains model loading and configuration functions.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    Load VGG16 with Batch Normalization pretrained on ImageNet.
    Replace the final classifier layer for the target number of classes.
    
    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze the feature extractor backbone.
        
    Returns:
        Modified VGG16-BN model.
    """
    print(f"[INFO] Loading VGG16-BN pretrained on ImageNet...")
    
    # Load pretrained VGG16 with Batch Normalization
    model = models.vgg16_bn(weights='IMAGENET1K_V1')
    
    # Freeze backbone if requested
    if freeze_backbone:
        print("[INFO] Freezing backbone layers...")
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Get the number of input features for the final classifier layer
    # VGG16 classifier: [Linear(25088, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Dropout, Linear(4096, 1000)]
    in_features = model.classifier[6].in_features  # 4096
    
    # Replace the final classifier layer
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    print(f"[INFO] Modified classifier: Linear({in_features}, {num_classes})")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def get_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for the model.
    
    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        
    Returns:
        AdamW optimizer.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    scheduler_type: str = 'cosine'
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        scheduler_type: Type of scheduler ('cosine', 'linear', 'step').
        
    Returns:
        Learning rate scheduler.
    """
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=num_training_steps // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def save_model(model: nn.Module, save_path: str, class_to_idx: dict = None) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save.
        save_path: Path to save the model.
        class_to_idx: Optional class to index mapping.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': 'vgg16_bn',
    }
    
    if class_to_idx:
        checkpoint['class_to_idx'] = class_to_idx
        checkpoint['idx_to_class'] = {v: k for k, v in class_to_idx.items()}
    
    torch.save(checkpoint, save_path)
    print(f"[INFO] Model saved to: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: The model architecture.
        checkpoint_path: Path to the checkpoint.
        
    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    return model
