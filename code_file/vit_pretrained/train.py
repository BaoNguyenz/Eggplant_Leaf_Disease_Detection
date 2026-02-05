"""
Main training script for ViT model.
Orchestrates the complete training pipeline with metrics tracking and visualization.

Author: Senior AI Engineer
Date: 2026-02-05
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import logging

from config import get_args, print_config
from dataset import create_dataloaders
from model import create_vit_model
from loss import get_loss_fn
from utils import (
    setup_logger,
    EarlyStopping,
    calculate_metrics,
    plot_loss_curves,
    plot_f1_curve,
    plot_confusion_matrix,
    save_checkpoint,
    load_checkpoint,
    save_training_history
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> tuple:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, y_true, y_pred)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_labels), np.array(all_preds)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    phase: str = 'Val'
) -> tuple:
    """
    Validate/test the model.
    
    Args:
        model: Model to validate
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        phase: Phase name ('Val' or 'Test')
    
    Returns:
        Tuple of (average_loss, y_true, y_pred)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{phase}]', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_labels), np.array(all_preds)


def main():
    """Main training function."""
    # Parse arguments
    args = get_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logger(args.output_dir)
    
    # Print configuration
    print_config(args)
    logger.info("Starting ViT training pipeline...")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Create dataloaders
    logger.info("\n" + "="*60)
    logger.info("LOADING DATA")
    logger.info("="*60)
    train_loader, val_loader, test_loader, class_weights, class_names = create_dataloaders(args)
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("BUILDING MODEL")
    logger.info("="*60)
    model = create_vit_model(
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
        gpu_ids=args.gpu_ids
    )
    
    # Create loss function
    logger.info("\n" + "="*60)
    logger.info("SETTING UP TRAINING")
    logger.info("="*60)
    criterion = get_loss_fn(
        loss_name=args.loss,
        class_weights=class_weights,
        device=args.device,
        focal_gamma=args.focal_gamma
    )
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        logger.info(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        logger.info(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Create learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        logger.info("Learning rate scheduler: ReduceLROnPlateau (monitoring val F1)")
    
    # Setup early stopping
    early_stopper = None
    if args.early_stopping > 0:
        early_stopper = EarlyStopping(patience=args.early_stopping, mode='max')
        logger.info(f"Early stopping enabled with patience={args.early_stopping}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("TRAINING")
    logger.info("="*60)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_labels, train_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        train_metrics = calculate_metrics(train_labels, train_preds)
        
        # Validate
        val_loss, val_labels, val_preds = validate(
            model, val_loader, criterion, args.device, epoch, phase='Val'
        )
        val_metrics = calculate_metrics(val_labels, val_preds)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                args.output_dir / 'best_model.pth'
            )
            logger.info(f"✓ Best model saved! (F1: {best_val_f1:.4f})")
        
        # Early stopping check
        if early_stopper is not None:
            early_stopper(val_metrics['f1'], epoch)
            if early_stopper.should_stop():
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                logger.info(f"Best F1: {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch}")
                break
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    model, _ = load_checkpoint(model, args.output_dir / 'best_model.pth', args.device)
    
    test_loss, test_labels, test_preds = validate(
        model, test_loader, criterion, args.device, epoch=0, phase='Test'
    )
    test_metrics = calculate_metrics(test_labels, test_preds)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
    
    # Save training history
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)
    save_training_history(history, args.output_dir / 'training_history.json')
    
    # Generate plots
    plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        args.output_dir / 'loss_curves.png'
    )
    
    plot_f1_curve(
        history['train_f1'],
        history['val_f1'],
        args.output_dir / 'f1_curve.png'
    )
    
    plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names,
        args.output_dir / 'confusion_matrix.png',
        normalize=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    # Windows multiprocessing guard
    main()
