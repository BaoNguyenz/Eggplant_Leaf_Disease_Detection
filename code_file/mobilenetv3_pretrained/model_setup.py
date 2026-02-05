"""
Model Setup Module for MobileNetV3 Small
Configures pretrained MobileNetV3 Small for custom classification task
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


def create_mobilenetv3_model(num_classes, pretrained=True, freeze_features=False):
    """
    Create MobileNetV3 Small model with custom classifier
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained weights
        freeze_features (bool): Whether to freeze feature extractor layers
        
    Returns:
        torch.nn.Module: MobileNetV3 Small model
    """
    
    # Load pretrained MobileNetV3 Small
    if pretrained:
        print("[INFO] Loading MobileNetV3 Small with ImageNet pretrained weights")
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        print("[INFO] Loading MobileNetV3 Small without pretrained weights")
        model = models.mobilenet_v3_small(weights=None)
    
    # Freeze feature extractor if specified
    if freeze_features:
        print("[INFO] Freezing feature extractor layers")
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        print("[INFO] Training all layers (no freezing)")
    
    # MobileNetV3 classifier structure:
    # classifier = Sequential(
    #     Linear(in_features=576, out_features=1024, bias=True),
    #     Hardswish(),
    #     Dropout(p=0.2, inplace=True),
    #     Linear(in_features=1024, out_features=1000, bias=True)
    # )
    
    # Get the input features of the last linear layer
    in_features = model.classifier[-1].in_features  # 1024
    
    # Replace the classifier
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    print(f"[INFO] Modified classifier output to {num_classes} classes")
    print(f"[INFO] Model architecture:")
    print(f"       - Feature extractor: MobileNetV3 Small features")
    print(f"       - Classifier input features: {in_features}")
    print(f"       - Classifier output classes: {num_classes}")
    
    return model


def count_parameters(model):
    """
    Count trainable and total parameters in the model
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Total parameters: {total_params:,}")
    
    return trainable_params, total_params


if __name__ == "__main__":
    # Test model creation
    print("Testing MobileNetV3 Small model creation...")
    model = create_mobilenetv3_model(num_classes=5, pretrained=True, freeze_features=False)
    count_parameters(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"[INFO] Output shape: {output.shape}")
    print("[INFO] Model test successful!")
