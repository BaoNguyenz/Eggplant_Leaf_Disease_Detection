"""
VGG16 Model Module
Defines VGG16 feature extractor with Global Average Pooling.
Supports loading custom fine-tuned weights.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
from pathlib import Path


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 Feature Extractor with Global Average Pooling.
    
    This model removes the fully connected layers and uses GAP to produce
    512-dimensional feature vectors suitable for traditional ML classifiers.
    """
    
    def __init__(self, pretrained: bool = True, custom_weights_path: Optional[str] = None):
        """
        Initialize VGG16 feature extractor.
        
        Args:
            pretrained: Whether to load ImageNet pre-trained weights (ignored if custom_weights_path is provided)
            custom_weights_path: Path to custom checkpoint file (.pth) from fine-tuned model
        """
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pre-trained VGG16 architecture
        if custom_weights_path is not None:
            # Load custom weights from fine-tuned model
            print(f"📂 Loading custom weights from checkpoint...")
            print(f"   Path: {custom_weights_path}")
            
            # Use VGG16-BN (with Batch Normalization) to match training architecture
            base_model = models.vgg16_bn(weights=None)  # Start with empty weights
            
            # Load checkpoint
            checkpoint = torch.load(custom_weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'epoch' in checkpoint:
                    print(f"   └─ Epoch: {checkpoint['epoch']}")
                if 'best_f1' in checkpoint:
                    print(f"   └─ Best F1: {checkpoint['best_f1']:.4f}")
                if 'model_architecture' in checkpoint:
                    print(f"   └─ Architecture: {checkpoint['model_architecture']}")
            else:
                state_dict = checkpoint
            
            # Load weights into base model
            # Filter out classifier weights if they exist (we only need features)
            features_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features.'):
                    features_state_dict[key] = value
            
            base_model.load_state_dict(features_state_dict, strict=False)
            print(f"   └─ Loaded {len(features_state_dict)} feature layer parameters ✓")
            
        elif pretrained:
            # Load ImageNet pre-trained weights
            print("📂 Loading ImageNet pretrained weights...")
            weights = models.VGG16_Weights.IMAGENET1K_V1
            base_model = models.vgg16(weights=weights)
        else:
            # No weights
            print("📂 Initializing VGG16 with random weights...")
            base_model = models.vgg16(weights=None)
        
        # Extract only the feature extraction layers (convolutional layers)
        self.features = base_model.features
        
        # Add Global Average Pooling to reduce spatial dimensions
        # Output: (batch_size, 512, 1, 1) -> squeeze to (batch_size, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze all layers for feature extraction (no training)
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze all model parameters to prevent training."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature tensor of shape (batch_size, 512)
        """
        # Pass through convolutional layers
        x = self.features(x)  # Output: (batch_size, 512, 7, 7)
        
        # Apply Global Average Pooling
        x = self.global_avg_pool(x)  # Output: (batch_size, 512, 1, 1)
        
        # Flatten to (batch_size, 512)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of extracted features.
        
        Returns:
            Feature dimension (512 for VGG16)
        """
        return 512


def create_feature_extractor(
    pretrained: bool = True, 
    device: Optional[str] = None,
    custom_weights_path: Optional[str] = None
) -> VGG16FeatureExtractor:
    """
    Factory function to create and configure feature extractor.
    
    Args:
        pretrained: Whether to use ImageNet pre-trained weights (ignored if custom_weights_path is provided)
        device: Device to load model on ('cuda' or 'cpu')
        custom_weights_path: Path to custom checkpoint file from fine-tuned model
        
    Returns:
        Configured VGG16FeatureExtractor model
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = VGG16FeatureExtractor(pretrained=pretrained, custom_weights_path=custom_weights_path)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    weights_source = "Custom fine-tuned weights" if custom_weights_path else ("ImageNet pretrained" if pretrained else "Random initialization")
    print(f"✓ VGG16 Feature Extractor loaded on {device}")
    print(f"✓ Weights source: {weights_source}")
    print(f"✓ Feature dimension: {model.get_feature_dim()}")
    
    return model
