"""
Model module for ResNet34 feature extraction with custom weights
"""
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet34


class ResNet34FeatureExtractor(nn.Module):
    """
    ResNet34 Feature Extractor with custom trained weights
    
    Architecture:
        - Load ResNet34 base architecture
        - Load custom weights from .pth file
        - Remove final FC layer
        - Use Global Average Pooling
        - Output: 512-dimensional feature vector
    """
    
    def __init__(self, weight_path: Optional[Path] = None, device: str = "cpu"):
        """
        Initialize ResNet34 feature extractor
        
        Args:
            weight_path: Path to custom .pth weights file (optional)
            device: Device to load model on ('cpu' or 'cuda')
        """
        super(ResNet34FeatureExtractor, self).__init__()
        
        self.device = device
        self.feature_dim = 512  # ResNet34 feature dimension
        
        # Initialize ResNet34 architecture (no pretrained weights initially)
        print("Initializing ResNet34 architecture...")
        self.backbone = resnet34(pretrained=False)
        
        # Load custom weights if provided
        if weight_path is not None:
            self._load_custom_weights(weight_path)
        else:
            print("⚠️  Warning: No custom weights provided. Using random initialization.")
        
        # Remove the final fully connected layer
        # ResNet architecture: ... -> avgpool -> fc
        # We keep everything except the fc layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # The output will be [batch_size, 512, 1, 1] after avgpool
        # We'll flatten it in the forward pass
        
        # Move model to device
        self.to(self.device)
        self.eval()  # Set to evaluation mode
        
        print(f"✓ Model initialized on {self.device}")
        print(f"✓ Feature dimension: {self.feature_dim}")
    
    def _load_custom_weights(self, weight_path: Path) -> None:
        """
        Load custom weights from .pth file
        
        Args:
            weight_path: Path to .pth checkpoint file
            
        Handles various checkpoint formats:
            - Direct state_dict
            - Checkpoint with 'state_dict' key
            - Checkpoint with 'model_state_dict' key
            - Keys with 'module.' prefix (from DataParallel)
        """
        weight_path = Path(weight_path)
        
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        print(f"Loading custom weights from: {weight_path}")
        
        try:
            # Load checkpoint (map to CPU first to avoid device issues)
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # Extract state_dict from checkpoint
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("  └─ Found 'state_dict' key in checkpoint")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("  └─ Found 'model_state_dict' key in checkpoint")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print("  └─ Found 'model' key in checkpoint")
                else:
                    # Assume the entire dict is the state_dict
                    state_dict = checkpoint
                    print("  └─ Using entire checkpoint as state_dict")
            else:
                state_dict = checkpoint
                print("  └─ Checkpoint is direct state_dict")
            
            # Remove 'module.' prefix if present (from DataParallel training)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            if any(k.startswith('module.') for k in state_dict.keys()):
                print("  └─ Removed 'module.' prefix from keys")
            
            # Filter out FC layer weights (we're doing feature extraction)
            # The checkpoint may have a custom FC layer (e.g., 6 classes)
            # but we only need the feature extraction layers
            filtered_state_dict = {}
            fc_keys_skipped = []
            
            for k, v in new_state_dict.items():
                # Skip FC layer weights
                if k.startswith('fc.'):
                    fc_keys_skipped.append(k)
                    continue
                filtered_state_dict[k] = v
            
            if fc_keys_skipped:
                print(f"  └─ Skipped FC layer weights: {fc_keys_skipped}")
                print(f"     (FC layer will be removed for feature extraction)")
            
            # Load weights with strict=False to handle any remaining mismatches
            missing_keys, unexpected_keys = self.backbone.load_state_dict(
                filtered_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"  └─ Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"     - {key}")
            
            if unexpected_keys:
                print(f"  └─ Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"     - {key}")
            
            print("✓ Custom weights loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weight_path}: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Feature tensor of shape [batch_size, 512]
        """
        # Pass through feature extractor (all layers except FC)
        # Output shape: [batch_size, 512, 1, 1]
        features = self.features(x)
        
        # Flatten to [batch_size, 512]
        features = torch.flatten(features, 1)
        
        return features
    
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without gradient computation
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Feature tensor of shape [batch_size, 512]
        """
        return self.forward(x)


if __name__ == "__main__":
    # Test model initialization
    from config import MODEL_WEIGHT_PATH, DEVICE
    
    print("\n" + "=" * 80)
    print("Testing ResNet34 Feature Extractor")
    print("=" * 80 + "\n")
    
    # Initialize model with custom weights
    model = ResNet34FeatureExtractor(
        weight_path=MODEL_WEIGHT_PATH,
        device=str(DEVICE)
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    features = model.extract_features(dummy_input)
    
    print(f"\nTest Input Shape : {dummy_input.shape}")
    print(f"Output Features  : {features.shape}")
    print(f"Feature Dimension: {features.shape[1]}")
    
    assert features.shape == (2, 512), "Output shape mismatch!"
    print("\n✓ Model test passed!")
