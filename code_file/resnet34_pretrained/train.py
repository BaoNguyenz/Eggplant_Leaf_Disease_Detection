"""
Main training script for ResNet34 on eggplant leaf disease detection.
Supports command-line arguments for flexible configuration.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from utils import (
    set_seed, get_class_weights, compute_metrics, check_dir,
    save_training_log, save_loss_f1_curves, save_confusion_matrix, FocalLoss
)
from data_setup import create_dataloaders
from model_setup import create_resnet34, get_model_info


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train ResNet34 for Eggplant Leaf Disease Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str,
                        default=r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default=r'E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\resnet34_pretrained',
                        help='Path to save model and logs')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal'],
                        help='Loss function type')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='Use multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (average_loss, f1_score)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
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
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics['f1_score']


def validate(model, dataloader, criterion, device):
    """
    Validate model.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average_loss, f1_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics['f1_score']


def test_model(model, dataloader, device, class_names):
    """
    Test model and return predictions and labels.
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to test on
        class_names: List of class names
        
    Returns:
        tuple: (all_labels, all_preds, metrics_dict)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("="*60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    return all_labels, all_preds, metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    check_dir(output_dir)
    
    print("\n" + "="*60)
    print("RESNET34 TRAINING - EGGPLANT LEAF DISEASE DETECTION")
    print("="*60)
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Num workers:      {args.num_workers}")
    print(f"Loss type:        {args.loss_type}")
    print(f"Patience:         {args.patience}")
    print(f"Random seed:      {args.seed}")
    print(f"Multi-GPU:        {args.use_multi_gpu}")
    if args.use_multi_gpu:
        print(f"GPU IDs:          {args.gpu_ids}")
    print("="*60 + "\n")
    
    # Setup device and multi-GPU configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_multi_gpu and torch.cuda.is_available():
        # Parse GPU IDs
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
        
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            print(f"✓ Multi-GPU training enabled with {num_gpus} GPUs")
            for i, gpu_id in enumerate(gpu_ids):
                print(f"  GPU {i}: {torch.cuda.get_device_name(gpu_id)}")
            device = torch.device(f'cuda:{gpu_ids[0]}')  # Primary GPU
        else:
            print(f"⚠ Warning: Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            print(f"  Falling back to single GPU: {torch.cuda.get_device_name(0)}")
            args.use_multi_gpu = False
    else:
        print(f"✓ Using device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names, full_dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    num_classes = len(class_names)
    
    # Create model
    print("\nCreating model...")
    model = create_resnet34(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Wrap model with DataParallel for multi-GPU
    if args.use_multi_gpu:
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"✓ Model wrapped with DataParallel (GPUs: {gpu_ids})")
    
    get_model_info(model)
    
    # Setup loss function
    class_weights = get_class_weights(full_dataset)
    class_weights = class_weights.to(device)
    
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✓ Using CrossEntropyLoss with class weights")
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"✓ Using FocalLoss (gamma=2.0) with class weights")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"✓ Using Adam optimizer (lr={args.lr})")
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_path = output_dir / 'best_model.pth'
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
            print(f"✓ Best model saved (Val F1: {val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            print(f"  Best Val F1: {best_val_f1:.4f}")
            break
        
        print()
    
    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60 + "\n")
    
    # Save training visualizations
    print("Saving training visualizations...")
    save_training_log(history, output_dir)
    save_loss_f1_curves(history, output_dir)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(best_model_path)
    
    # Load into the underlying model (handle DataParallel)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    all_labels, all_preds, test_metrics = test_model(model, test_loader, device, class_names)
    
    # Save confusion matrix
    print("\nSaving confusion matrix...")
    save_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - best_model.pth")
    print(f"  - training_log.csv")
    print(f"  - training_curves.png")
    print(f"  - confusion_matrix.png")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
