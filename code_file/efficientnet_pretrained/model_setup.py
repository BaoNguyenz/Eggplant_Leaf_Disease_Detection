"""
Model setup for EfficientNet-B0 with ImageNet pretrained weights.
Native input resolution: 224x224.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create EfficientNet-B0 model with custom classifier.
    
    EfficientNet-B0 Details:
        - Native input size: 224x224
        - Feature dimension: 1280
        - Parameters: ~5.3M
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Modified EfficientNet-B0 model
    """
    print("\n" + "="*60)
    print("MODEL SETUP: EfficientNet-B0")
    print("="*60)
    
    # Load EfficientNet-B0 with default (ImageNet) weights
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        print("✓ Loaded ImageNet pretrained weights")
    else:
        model = models.efficientnet_b0(weights=None)
        print("✓ Initialized random weights")
    
    # EfficientNet-B0 architecture info
    print(f"✓ Native input size: 224x224")
    print(f"✓ Feature dimension: 1280")
    
    # Modify classifier for custom number of classes
    # Original classifier: Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    print(f"✓ Modified classifier: {in_features} → {num_classes} classes")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")
    
    return model


# Alias for backward compatibility
create_efficientnet_b7 = create_efficientnet_b0


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze the backbone (feature extractor) of EfficientNet-B0.
    
    Args:
        model: EfficientNet-B7 model
        freeze: If True, freeze backbone; if False, unfreeze
    """
    # Freeze/unfreeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = not freeze
    
    # Always keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    if freeze:
        print("✓ Backbone frozen (feature extraction mode)")
    else:
        print("✓ Backbone unfrozen (fine-tuning mode)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}\n")


def get_model_summary(model: nn.Module) -> None:
    """
    Print detailed model summary.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(model)
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNet-B0 Model Setup...")
    
    # Create model with 6 classes (example)
    model = create_efficientnet_b0(num_classes=6, pretrained=True)
    
    # Test forward pass with 224x224 input
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model test passed!")
