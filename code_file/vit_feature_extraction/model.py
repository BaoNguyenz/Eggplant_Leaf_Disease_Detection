"""
Model module for Vision Transformer (ViT) feature extraction.
Extracts 768-dimensional features from the [CLS] token.
"""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT-B/16) Feature Extractor.
    
    Extracts 768-dimensional feature vectors from the [CLS] token
    at the final encoder layer, following the standard ViT architecture.
    
    Architecture:
        - Input: RGB images (224x224)
        - Patch embedding: 16x16 patches
        - Encoder: 12 transformer blocks
        - Output: [CLS] token embedding (768 dimensions)
    
    Reference:
        Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers 
        for Image Recognition at Scale." ICLR 2021.
    """
    
    def __init__(self, pretrained: bool = True, custom_weights_path: str = None):
        """
        Initialize ViT feature extractor.
        
        Args:
            pretrained: Whether to load ImageNet-1K pretrained weights (ignored if custom_weights_path is provided)
            custom_weights_path: Path to custom trained model checkpoint (.pth file)
                               If provided, will load this instead of ImageNet weights
        """
        super(ViTFeatureExtractor, self).__init__()
        
        # Load base ViT-B/16 architecture
        if custom_weights_path:
            # Load custom trained weights from checkpoint
            print(f"[INFO] Loading custom ViT weights from: {custom_weights_path}")
            
            # Initialize model without pretrained weights first
            self.vit = vit_b_16(weights=None)
            
            # Load checkpoint
            checkpoint = torch.load(custom_weights_path, map_location='cpu', weights_only=False)
            
            # Extract model state dict (handle different checkpoint formats)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"[INFO] Checkpoint metrics: {checkpoint['metrics']}")
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if model was saved with DataParallel/DistributedDataParallel
            # This happens when model is wrapped with nn.DataParallel during training
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    # Remove 'module.' prefix
                    new_key = key[7:]  # len('module.') = 7
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Remove classification head weights (heads.head.*) as they are task-specific
            # We only need the feature encoder weights for feature extraction
            filtered_state_dict = {
                k: v for k, v in new_state_dict.items() 
                if not k.startswith('heads.')
            }
            
            print(f"[INFO] Loading {len(filtered_state_dict)} weights (excluding classification head)")
            
            # Load weights into model (strict=False to allow missing classification head)
            missing_keys, unexpected_keys = self.vit.load_state_dict(filtered_state_dict, strict=False)
            
            # Check if only classification head keys are missing (expected)
            expected_missing = all('heads' in key for key in missing_keys)
            if not expected_missing:
                print(f"[WARNING] Unexpected missing keys: {[k for k in missing_keys if 'heads' not in k]}")
            
            print("[SUCCESS] Custom weights loaded successfully!")
            
        elif pretrained:
            # Load ImageNet pretrained weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=weights)
            print(f"[INFO] Loaded ViT-B/16 with {weights.name} pretrained weights")
        else:
            # Load without any pretrained weights
            self.vit = vit_b_16(weights=None)
            print("[INFO] Loaded ViT-B/16 without pretrained weights")
        
        # Set model to evaluation mode (disable dropout, batch norm training mode)
        self.vit.eval()
        
        # Feature dimension (output of [CLS] token)
        self.feature_dim = 768
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Feature tensor of shape [batch_size, 768]
            
        Processing steps:
            1. Patch embedding: Convert image to sequence of patches
            2. Add [CLS] token at position 0
            3. Add positional embeddings
            4. Pass through transformer encoder (12 blocks)
            5. Extract [CLS] token from encoder output
        """
        # Input: [batch_size, 3, 224, 224]
        batch_size = x.shape[0]
        
        # Expand the class token to the batch size
        # [CLS] token is a learnable embedding added to the beginning of the sequence
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)  # [batch_size, 1, 768]
        
        # Patch embedding: Convert 224x224 image to 196 patches (14x14 grid of 16x16 patches)
        x = self.vit.conv_proj(x)  # [batch_size, 768, 14, 14]
        x = x.flatten(2)  # [batch_size, 768, 196]
        x = x.transpose(1, 2)  # [batch_size, 196, 768]
        
        # Concatenate [CLS] token at the beginning
        # Result: [batch_size, 197, 768] where 197 = 1 (CLS) + 196 (patches)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embeddings (learned during pre-training)
        # These embeddings help the model understand spatial relationships between patches
        x = x + self.vit.encoder.pos_embedding
        
        # Apply dropout (disabled in eval mode)
        x = self.vit.encoder.dropout(x)
        
        # Pass through transformer encoder (12 blocks of multi-head self-attention)
        # Each block allows patches to interact and build contextual representations
        x = self.vit.encoder.layers(x)  # [batch_size, 197, 768]
        
        # Apply layer normalization
        x = self.vit.encoder.ln(x)  # [batch_size, 197, 768]
        
        # ========================================================================
        # CRITICAL: Extract [CLS] token (position 0) as the image representation
        # ========================================================================
        # Why x[:, 0]?
        # - Position 0 contains the [CLS] (classification) token
        # - During pre-training, only this token is fed to the classification head
        # - It aggregates information from all patches via self-attention
        # - This is the standard approach in the original ViT paper
        # - Pre-trained weights are optimized for this specific representation
        # 
        # Alternative approaches (NOT used here):
        # - x[:, 1:].mean(dim=1): Average all patch tokens (ignores pre-training)
        # - x.mean(dim=1): Average including CLS (non-standard)
        # 
        # Output shape: [batch_size, 768]
        # ========================================================================
        features = x[:, 0]
        
        return features
    
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without gradient computation (for inference).
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Feature tensor of shape [batch_size, 768]
        """
        return self.forward(x)


def create_feature_extractor(
    device: torch.device,
    custom_weights_path: str = None,
    use_pretrained: bool = True
) -> ViTFeatureExtractor:
    """
    Create and initialize ViT feature extractor.
    
    Args:
        device: Computing device (cuda or cpu)
        custom_weights_path: Optional path to custom trained weights (.pth file)
                           If provided, this takes priority over pretrained weights
        use_pretrained: Whether to use ImageNet pretrained weights (ignored if custom_weights_path is set)
        
    Returns:
        ViTFeatureExtractor model moved to specified device
    """
    model = ViTFeatureExtractor(
        pretrained=use_pretrained,
        custom_weights_path=custom_weights_path
    )
    model = model.to(device)
    model.eval()  # Ensure evaluation mode
    
    print(f"[INFO] ViT feature extractor ready on {device}")
    print(f"[INFO] Feature dimension: {model.feature_dim}")
    
    return model
