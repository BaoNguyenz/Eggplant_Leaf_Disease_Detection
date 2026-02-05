"""
Model architecture setup for ResNet34.
Loads pretrained ResNet34 and adapts it for custom number of classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_resnet34(num_classes, pretrained=True):
    """
    Create ResNet34 model with custom number of output classes.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained weights
        
    Returns:
        nn.Module: ResNet34 model adapted for num_classes
    """
    # Load pretrained ResNet34
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        print("✓ Loaded ResNet34 with ImageNet pretrained weights")
    else:
        model = models.resnet34(weights=None)
        print("✓ Loaded ResNet34 without pretrained weights")
    
    # Get the number of input features for the final fully connected layer
    num_features = model.fc.in_features
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(num_features, num_classes)
    
    print(f"✓ Adapted ResNet34 for {num_classes} classes")
    print(f"  Final layer: Linear(in_features={num_features}, out_features={num_classes})")
    
    return model


def get_model_info(model):
    """
    Print model information including total parameters.
    
    Args:
        model (nn.Module): PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Model Information ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"-------------------------\n")
