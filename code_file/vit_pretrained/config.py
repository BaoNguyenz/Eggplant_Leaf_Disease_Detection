"""
Configuration module for ViT training pipeline.
Manages all hyperparameters and settings via command-line arguments.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import argparse
import torch
from pathlib import Path
from typing import Any


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ViT training.
    
    Returns:
        argparse.Namespace: Parsed arguments with all training configurations
    """
    parser = argparse.ArgumentParser(
        description='Vision Transformer Training for Eggplant Leaf Disease Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images',
        help='Path to dataset directory with class subdirectories'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Proportion of data for training (0.0-1.0)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Proportion of data for validation (0.0-1.0)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training and validation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay (L2 regularization)'
    )
    
    # Model parameters
    parser.add_argument(
        '--num_classes',
        type=int,
        default=6,
        help='Number of disease classes'
    )
    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='Freeze ViT encoder and only train classification head'
    )
    
    # Loss function
    parser.add_argument(
        '--loss',
        type=str,
        default='cross_entropy',
        choices=['cross_entropy', 'focal_loss'],
        help='Loss function to use'
    )
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Gamma parameter for Focal Loss (focusing parameter)'
    )
    
    # Optimizer
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adam', 'adamw'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use ReduceLROnPlateau learning rate scheduler'
    )
    
    # Early stopping
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=10,
        help='Early stopping patience (epochs). Set to 0 to disable'
    )
    
    # System parameters
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (set to 0 on Windows if issues)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: "cuda", "cpu", or "auto" for automatic detection'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='GPU IDs to use (e.g., "0" for single GPU, "0,1" for multi-GPU)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory to save outputs (models, plots, logs)'
    )
    
    # Augmentation
    parser.add_argument(
        '--use_weighted_sampler',
        action='store_true',
        help='Use WeightedRandomSampler for training to oversample minority classes'
    )
    
    args = parser.parse_args()
    
    # Post-processing
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    
    # Auto-detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse GPU IDs
    if args.device == 'cuda':
        args.gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
    else:
        args.gpu_ids = []
    
    # Validate splits
    if args.train_split + args.val_split >= 1.0:
        raise ValueError(
            f"train_split ({args.train_split}) + val_split ({args.val_split}) must be < 1.0"
        )
    
    return args


def print_config(args: argparse.Namespace) -> None:
    """
    Print configuration in a formatted way.
    
    Args:
        args: Parsed arguments
    """
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Data Directory:       {args.data_dir}")
    print(f"Output Directory:     {args.output_dir}")
    print(f"Data Split:           Train={args.train_split}, Val={args.val_split}, Test={1-args.train_split-args.val_split}")
    print(f"Device:               {args.device}")
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        print(f"GPU IDs:              {args.gpu_ids} (Multi-GPU: {len(args.gpu_ids) > 1})")
    print(f"Random Seed:          {args.seed}")
    print("-"*60)
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Weight Decay:         {args.weight_decay}")
    print(f"Optimizer:            {args.optimizer}")
    print(f"Use Scheduler:        {args.use_scheduler}")
    print("-"*60)
    print(f"Loss Function:        {args.loss}")
    if args.loss == 'focal_loss':
        print(f"Focal Gamma:          {args.focal_gamma}")
    print(f"Early Stopping:       {args.early_stopping if args.early_stopping > 0 else 'Disabled'}")
    print("-"*60)
    print(f"Num Classes:          {args.num_classes}")
    print(f"Freeze Backbone:      {args.freeze_backbone}")
    print(f"Weighted Sampler:     {args.use_weighted_sampler}")
    print(f"Num Workers:          {args.num_workers}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test configuration parsing
    args = get_args()
    print_config(args)
