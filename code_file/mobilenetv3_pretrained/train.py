"""
Main Training Script for MobileNetV3 Small
Eggplant Leaf Disease Detection with customizable hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# Import custom modules
from utils import (
    set_seed, get_class_weights, compute_metrics, check_dir,
    save_training_log, save_loss_f1_curves, save_confusion_matrix, FocalLoss
)
from data_setup import create_dataloaders
from model_setup import create_mobilenetv3_model, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (avg_loss, avg_f1)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics['f1_score']


def validate(model, dataloader, criterion, device):
    """
    Validate model
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (avg_loss, avg_f1, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics['f1_score'], all_preds, all_labels


def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    check_dir(output_dir)
    
    # Device configuration and multi-GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_multi_gpu and torch.cuda.is_available():
        # Parse GPU IDs
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
        
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            print(f"\n[INFO] Multi-GPU training enabled with {num_gpus} GPUs")
            for i, gpu_id in enumerate(gpu_ids):
                print(f"[INFO]   GPU {i}: {torch.cuda.get_device_name(gpu_id)}")
            device = torch.device(f'cuda:{gpu_ids[0]}')  # Primary GPU
        else:
            print(f"\n[WARNING] Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            print(f"[INFO] Falling back to single GPU: {torch.cuda.get_device_name(0)}")
            args.use_multi_gpu = False
    else:
        print(f"\n[INFO] Using device: {device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_loader, val_loader, test_loader, full_dataset, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    num_classes = len(class_names)
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    print(f"[INFO] Creating MobileNetV3 Small model...")
    model = create_mobilenetv3_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_features=False
    )
    model = model.to(device)
    
    # Wrap model with DataParallel for multi-GPU
    if args.use_multi_gpu:
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[INFO] Model wrapped with DataParallel (GPUs: {gpu_ids})")
    
    count_parameters(model)
    
    # Setup loss function
    print("\n" + "="*60)
    print("CONFIGURING TRAINING")
    print("="*60)
    class_weights = get_class_weights(full_dataset).to(device)
    
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"[INFO] Using Focal Loss (gamma=2.0) with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Using Cross Entropy Loss with class weights")
    
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(f"[INFO] Optimizer: Adam (lr={args.lr})")
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"[INFO] Optimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        print(f"[INFO] Optimizer: SGD (lr={args.lr}, momentum=0.9, weight_decay=1e-4)")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }
    
    # Early stopping variables
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pth'
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"[INFO] Total epochs: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Early stopping patience: {args.patience}")
    print()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        
        # Check for best model (based on validation F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the underlying model (unwrap DataParallel if used)
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'class_names': class_names
            }, best_model_path)
            print(f"✓ Best model saved! (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{args.patience})")
        
        print()
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch} epochs")
            break
    
    # Post-training evaluation on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # Load into the underlying model (handle DataParallel)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']}")
    print(f"[INFO] Best Val F1: {checkpoint['val_f1']:.4f}")
    
    # Evaluate on test set
    test_loss, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Generate visualizations and reports
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save training log
    save_training_log(history, output_dir)
    
    # Save training curves
    save_loss_f1_curves(history, output_dir)
    
    # Save confusion matrix
    save_confusion_matrix(test_labels, test_preds, class_names, output_dir)
    
    # Save classification report
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"[INFO] All outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train MobileNetV3 Small for Eggplant Leaf Disease Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_pretrained',
        help='Path to save outputs (model, logs, plots)'
    )
    
    # Training arguments
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
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Loss function
    parser.add_argument(
        '--loss_type',
        type=str,
        choices=['cross_entropy', 'focal'],
        default='focal',
        help='Loss function type'
    )
    
    # Optimizer
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd'],
        default='adam',
        help='Optimizer type (adam, adamw, or sgd)'
    )
    
    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs)'
    )
    
    # Multi-GPU arguments
    parser.add_argument(
        '--use_multi_gpu',
        action='store_true',
        help='Use multiple GPUs with DataParallel'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0,1',
        help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:15s}: {value}")
    
    # Run training
    main(args)
