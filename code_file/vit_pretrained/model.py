"""
Model module for ViT training pipeline.
Creates and configures Vision Transformer model with transfer learning.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import logging

logger = logging.getLogger(__name__)


def create_vit_model(
    num_classes: int = 6,
    freeze_backbone: bool = False,
    device: str = 'cuda',
    gpu_ids: list = [0]
) -> nn.Module:
    """
    Create ViT-B/16 model with ImageNet pretrained weights.
    
    Args:
        num_classes: Number of output classes
        freeze_backbone: If True, freeze encoder and only train classification head
        device: Device to place model on
        gpu_ids: List of GPU IDs for multi-GPU training
    
    Returns:
        Configured ViT model (wrapped with DataParallel if multi-GPU)
    """
    logger.info("Loading ViT-B/16 model with ImageNet pretrained weights...")
    
    # Load model with ImageNet weights
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Freeze backbone if requested
    if freeze_backbone:
        logger.info("Freezing ViT encoder (backbone)...")
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classification head
    # ViT-B/16 has 768-dimensional embeddings
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    logger.info(f"Replaced classification head: {in_features} -> {num_classes}")
    
    # Move to device
    model = model.to(device)
    
    # Multi-GPU support using DataParallel
    if device == 'cuda' and len(gpu_ids) > 1:
        logger.info(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"Model wrapped with DataParallel across {len(gpu_ids)} GPUs")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"  Device: {device}")
    if device == 'cuda' and len(gpu_ids) > 1:
        logger.info(f"  Multi-GPU: True ({len(gpu_ids)} GPUs)")
    
    return model


if __name__ == '__main__':
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n=== Testing ViT Model Creation ===\n")
    
    # Test with trainable backbone
    print("1. Creating model with trainable backbone...")
    model = create_vit_model(num_classes=6, freeze_backbone=False, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (2, 6), f"Expected (2, 6), got {output.shape}"
    print("   ✓ Forward pass successful!\n")
    
    # Test with frozen backbone
    print("2. Creating model with frozen backbone...")
    model_frozen = create_vit_model(num_classes=6, freeze_backbone=True, device=device)
    print("   ✓ Frozen model created successfully!\n")
