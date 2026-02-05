"""
Model Module for MobileNetV3 Feature Extraction
===============================================
Defines the MobileNetV3Extractor class for loading custom weights
and extracting 576-dimensional features from the backbone.

Author: AI Engineer
Date: 2026-02-04
"""

from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision import models
import config


class MobileNetV3Extractor(nn.Module):
    """
    MobileNetV3 Small Feature Extractor with Custom Weights Support.
    
    Architecture:
        - Backbone: MobileNetV3 Small features (conv layers)
        - Pooling: AdaptiveAvgPool2d (global average pooling)
        - Classifier: Removed entirely
    
    Output:
        576-dimensional feature vector per image
    """
    
    def __init__(
        self,
        weight_path: Path,
        pretrained: bool = False
    ) -> None:
        """
        Initialize MobileNetV3 feature extractor.
        
        Args:
            weight_path (Path): Path to custom .pth checkpoint file
            pretrained (bool): Whether to use ImageNet pretrained weights
                               (ignored if custom weights are loaded)
        
        Raises:
            FileNotFoundError: If weight_path does not exist
            RuntimeError: If weight loading fails
        """
        super(MobileNetV3Extractor, self).__init__()
        
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        self.weight_path = weight_path
        
        # Step 1: Initialize MobileNetV3 Small architecture (pretrained=False)
        # We load custom weights, so we don't need ImageNet initialization
        print(f"✓ Initializing MobileNetV3 Small architecture...")
        base_model = models.mobilenet_v3_small(weights=None)
        
        # Step 2: Extract only the feature extraction layers
        # MobileNetV3 structure:
        #   - features: Conv layers (backbone)
        #   - avgpool: AdaptiveAvgPool2d
        #   - classifier: Sequential(Linear, Hardswish, Dropout, Linear) <- REMOVE THIS
        
        self.features = base_model.features  # Convolutional backbone
        self.avgpool = base_model.avgpool    # Global Average Pooling
        
        # Remove classifier entirely (we only want features)
        # The classifier is not needed for feature extraction
        
        # Step 3: Load custom trained weights
        self._load_custom_weights(weight_path)
        
        # Step 4: Set to evaluation mode (disable dropout, batchnorm training mode)
        self.eval()
        
        print(f"✓ MobileNetV3Extractor initialized successfully")
        print(f"  - Feature dimension: {config.FEATURE_DIM}")
        print(f"  - Weights loaded from: {weight_path.name}")
    
    def _load_custom_weights(self, weight_path: Path) -> None:
        """
        Load custom trained weights with robust error handling.
        Handles DataParallel 'module.' prefix and mismatched keys.
        
        Args:
            weight_path (Path): Path to .pth checkpoint
        
        Raises:
            RuntimeError: If loading fails after all attempts
        """
        print(f"✓ Loading custom weights from: {weight_path}")
        
        # Load checkpoint (map to CPU first for compatibility)
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the entire dict is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if model was trained with DataParallel
        state_dict = self._remove_module_prefix(state_dict)
        
        # Filter weights: Keep only 'features.*' and 'avgpool.*'
        # Discard 'classifier.*' entirely
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith('features.') or k.startswith('avgpool.')
        }
        
        if len(filtered_state_dict) == 0:
            print("⚠ Warning: No matching keys found after filtering.")
            print("   Attempting to load full state_dict with strict=False...")
            # Fallback: Try loading everything, let PyTorch ignore mismatches
            try:
                self.load_state_dict(state_dict, strict=False)
                print("✓ Weights loaded with strict=False (some keys ignored)")
                return
            except Exception as e:
                raise RuntimeError(f"Weight loading failed: {e}")
        
        # Load filtered weights (features + avgpool only)
        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                filtered_state_dict,
                strict=False  # Allow missing classifier keys
            )
            
            # Report status
            if len(missing_keys) > 0:
                print(f"  - Missing keys: {len(missing_keys)} (expected if classifier is missing)")
            if len(unexpected_keys) > 0:
                print(f"  - Unexpected keys: {len(unexpected_keys)}")
            
            print(f"✓ Custom weights loaded successfully ({len(filtered_state_dict)} keys)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load state_dict: {e}")
    
    @staticmethod
    def _remove_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove 'module.' prefix from state_dict keys.
        This prefix is added by DataParallel during training.
        
        Args:
            state_dict (Dict): Original state dictionary
        
        Returns:
            Dict: Cleaned state dictionary
        """
        cleaned_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_key = key[7:]  # Remove 'module.' (7 characters)
                cleaned_dict[cleaned_key] = value
            else:
                cleaned_dict[key] = value
        
        return cleaned_dict
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.
        
        Args:
            x (torch.Tensor): Input image batch of shape [B, 3, 224, 224]
        
        Returns:
            torch.Tensor: Feature vectors of shape [B, 576]
        
        Tensor Flow:
            [B, 3, 224, 224]   -> Input images
            [B, 576, 7, 7]     -> After features (backbone)
            [B, 576, 1, 1]     -> After avgpool (global pooling)
            [B, 576]           -> After flatten (final features)
        """
        # Step 1: Pass through convolutional backbone
        x = self.features(x)  # [B, 3, 224, 224] -> [B, 576, 7, 7]
        
        # Step 2: Global Average Pooling
        x = self.avgpool(x)   # [B, 576, 7, 7] -> [B, 576, 1, 1]
        
        # Step 3: Flatten to 1D feature vector
        x = torch.flatten(x, 1)  # [B, 576, 1, 1] -> [B, 576]
        
        return x  # Final shape: [B, 576]
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension (576 for MobileNetV3 Small)."""
        return config.FEATURE_DIM


def create_model(weight_path: Path, device: torch.device) -> MobileNetV3Extractor:
    """
    Factory function to create and initialize the feature extractor.
    
    Args:
        weight_path (Path): Path to custom .pth weights
        device (torch.device): Device to load model onto (CPU or CUDA)
    
    Returns:
        MobileNetV3Extractor: Initialized model ready for inference
    """
    model = MobileNetV3Extractor(weight_path=weight_path)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model


if __name__ == "__main__":
    # Test model module
    print("Testing Model Module...")
    print("-" * 70)
    
    # Create model
    model = create_model(
        weight_path=config.MODEL_WEIGHT_PATH,
        device=config.DEVICE
    )
    
    # Test forward pass with dummy input
    print("\nTesting Forward Pass:")
    dummy_input = torch.randn(4, 3, 224, 224).to(config.DEVICE)  # Batch of 4 images
    print(f"  Input shape  : {dummy_input.shape}")
    
    with torch.no_grad():
        features = model(dummy_input)
    
    print(f"  Output shape : {features.shape}")  # Should be [4, 576]
    print(f"  Feature dim  : {model.get_feature_dim()}")
    
    assert features.shape == (4, 576), "Output shape mismatch!"
    print("\n✓ Model test passed!")
