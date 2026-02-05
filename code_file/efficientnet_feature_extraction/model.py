"""
Model Module for EfficientNet-B0 Feature Extraction
===================================================
Định nghĩa class trích xuất đặc trưng dựa trên EfficientNet-B0.
Chỉ sử dụng phần convolution features, loại bỏ classifier.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Tuple, Optional
from pathlib import Path


class EfficientNetB0FeatureExtractor(nn.Module):
    """
    Feature Extractor sử dụng EfficientNet-B0 pretrained.
    
    Architecture:
    - Backbone: EfficientNet-B0 features (convolution blocks)
    - Pooling: Global Average Pooling (AdaptiveAvgPool2d)
    - Output: Feature vector 1280-dim
    
    Attributes:
        features: Convolutional feature extractor từ EfficientNet-B0
        global_pool: Global Average Pooling layer
        feature_dim: Output feature dimension (1280)
    """
    
    def __init__(self, pretrained: bool = True, custom_weights_path: Optional[Path] = None) -> None:
        """
        Khởi tạo Feature Extractor.
        
        Args:
            pretrained: Sử dụng pretrained weights từ ImageNet (ignored if custom_weights_path provided)
            custom_weights_path: Đường dẫn tới custom trained weights (.pth file)
        """
        super(EfficientNetB0FeatureExtractor, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if custom_weights_path:
            # Load custom weights
            weights = None
            print(f"✓ Loading EfficientNet-B0 with custom weights: {custom_weights_path}")
        elif pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            print(f"✓ Loading EfficientNet-B0 with pretrained weights: {weights}")
        else:
            weights = None
            print("⚠ Loading EfficientNet-B0 WITHOUT pretrained weights")
        
        # Load full model
        efficientnet = efficientnet_b0(weights=weights)
        
        # Chỉ lấy phần features (convolution blocks)
        # EfficientNet-B0 structure:
        #   - efficientnet.features: Tất cả convolution blocks
        #   - efficientnet.avgpool: Adaptive avg pooling (sẽ thay thế)
        #   - efficientnet.classifier: Fully connected layer (loại bỏ)
        self.features = efficientnet.features
        
        # Thay thế bằng Global Average Pooling tùy chỉnh
        # Output: [batch_size, 1280, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Feature dimension của EfficientNet-B0
        self.feature_dim = 1280
        
        # Load custom weights if provided
        if custom_weights_path:
            self._load_custom_weights(custom_weights_path)
        
        # Set model to evaluation mode
        self.eval()
        
        print(f"✓ EfficientNet-B0 Feature Extractor initialized")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Total parameters: {self._count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass để trích xuất features.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        
        Returns:
            Feature tensor [batch_size, 1280]
        """
        # 1. Pass through convolution blocks
        # Output shape: [batch_size, 1280, H', W'] (H', W' depend on input)
        x = self.features(x)
        
        # 2. Global Average Pooling
        # Output shape: [batch_size, 1280, 1, 1]
        x = self.global_pool(x)
        
        # 3. Flatten to [batch_size, 1280]
        x = torch.flatten(x, start_dim=1)
        
        return x
    
    def _count_parameters(self) -> int:
        """
        Đếm số lượng parameters trong model.
        
        Returns:
            Tổng số parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_feature_dim(self) -> int:
        """
        Trả về số chiều feature vector.
        
        Returns:
            Feature dimension (1280)
        """
        return self.feature_dim
    
    def _load_custom_weights(self, checkpoint_path: Path) -> None:
        """
        Load custom trained weights từ checkpoint.
        
        Xử lý:
        1. Load checkpoint dictionary
        2. Extract model_state_dict
        3. Remove 'module.' prefix (từ DataParallel)
        4. Filter chỉ lấy 'features.*' keys
        5. Load vào model
        
        Args:
            checkpoint_path: Đường dẫn tới file .pth checkpoint
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"  → Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  → Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, Best F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix and filter features only
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'module.' prefix (from DataParallel)
            new_key = key.replace('module.', '')
            
            # Only keep 'features.*' keys (skip classifier, fc, etc.)
            if new_key.startswith('features.'):
                new_state_dict[new_key] = value
        
        print(f"  → Filtered {len(new_state_dict)} feature layer parameters")
        
        # Load state dict (strict=False vì chỉ load features, không có classifier)
        missing_keys, unexpected_keys = self.features.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")
        
        print(f"  ✓ Custom weights loaded successfully")


def load_feature_extractor(
    device: torch.device, 
    pretrained: bool = True, 
    custom_weights_path: Optional[Path] = None
) -> EfficientNetB0FeatureExtractor:
    """
    Factory function để load model và chuyển lên device.
    
    Args:
        device: torch.device (cuda hoặc cpu)
        pretrained: Sử dụng pretrained weights (ignored if custom_weights_path provided)
        custom_weights_path: Đường dẫn tới custom trained weights
    
    Returns:
        EfficientNetB0FeatureExtractor instance trên device
    """
    model = EfficientNetB0FeatureExtractor(
        pretrained=pretrained, 
        custom_weights_path=custom_weights_path
    )
    model = model.to(device)
    
    # Set to evaluation mode (disable dropout, batchnorm training)
    model.eval()
    
    print(f"✓ Model moved to device: {device}")
    return model


def verify_output_shape(model: EfficientNetB0FeatureExtractor, input_size: Tuple[int, int] = (224, 224)) -> None:
    """
    Kiểm tra output shape của model với dummy input.
    
    Args:
        model: Feature extractor model
        input_size: Input image size (H, W)
    """
    # Create dummy input [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    dummy_input = dummy_input.to(next(model.parameters()).device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n========================================")
    print(f"Model Output Shape Verification")
    print(f"========================================")
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Expected     : torch.Size([1, 1280])")
    print(f"========================================\n")
    
    assert output.shape == (1, 1280), f"Output shape mismatch! Got {output.shape}"
    print(f"✓ Output shape verification PASSED")
