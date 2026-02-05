"""
Loss functions module for ViT training pipeline.
Implements CrossEntropyLoss with class weights and Focal Loss.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (2017)
        Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (Tensor of shape [num_classes])
        gamma: Focusing parameter (default=2.0). Higher gamma increases focus on hard examples.
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]
        
        Returns:
            Computed loss
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = probability of true class
        
        # Apply focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_fn(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of loss function ('cross_entropy' or 'focal_loss')
        class_weights: Tensor of class weights for handling imbalance
        device: Device to place weights on
        focal_gamma: Gamma parameter for Focal Loss
    
    Returns:
        Loss function module
    """
    # Move weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
    
    if loss_name == 'cross_entropy':
        logger.info("Loss function: CrossEntropyLoss with class weights")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
    elif loss_name == 'focal_loss':
        logger.info(f"Loss function: FocalLoss (gamma={focal_gamma}) with class weights")
        loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
        
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_fn


if __name__ == '__main__':
    # Test loss functions
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Loss Functions ===\n")
    
    # Create dummy data
    batch_size = 8
    num_classes = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dummy predictions and labels
    predictions = torch.randn(batch_size, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Dummy class weights
    class_weights = torch.tensor([1.5, 1.2, 1.0, 1.3, 1.8, 1.1]).to(device)
    
    print("1. Testing CrossEntropyLoss...")
    ce_loss = get_loss_fn('cross_entropy', class_weights, device)
    ce_output = ce_loss(predictions, labels)
    print(f"   Loss value: {ce_output.item():.4f}")
    print(f"   ✓ CrossEntropyLoss works!\n")
    
    print("2. Testing FocalLoss (gamma=2.0)...")
    focal_loss = get_loss_fn('focal_loss', class_weights, device, focal_gamma=2.0)
    focal_output = focal_loss(predictions, labels)
    print(f"   Loss value: {focal_output.item():.4f}")
    print(f"   ✓ FocalLoss works!\n")
    
    print("3. Testing FocalLoss (gamma=1.0) - should be similar to CE...")
    focal_loss_1 = get_loss_fn('focal_loss', class_weights, device, focal_gamma=1.0)
    focal_output_1 = focal_loss_1(predictions, labels)
    print(f"   Loss value: {focal_output_1.item():.4f}")
    print(f"   ✓ FocalLoss (gamma=1.0) works!\n")
    
    print("4. Testing without class weights...")
    ce_loss_no_weight = get_loss_fn('cross_entropy', None, device)
    ce_output_no_weight = ce_loss_no_weight(predictions, labels)
    print(f"   Loss value: {ce_output_no_weight.item():.4f}")
    print(f"   ✓ Loss without weights works!\n")
    
    print("All tests passed! ✓")
