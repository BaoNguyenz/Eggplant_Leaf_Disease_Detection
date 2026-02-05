"""
Main training script for EfficientNet-B0 on Eggplant Leaf Disease Detection.
Input resolution: 224x224 (native for EfficientNet-B0).

Usage:
    python train.py --epochs 50 --batch_size 32 --lr 0.0001 --patience 10
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

# Import custom modules
from data_setup import create_dataloaders
from model_setup import create_efficientnet_b0
from utils import (
    check_dir, get_class_weights, compute_metrics,
    save_loss_f1_curves, save_confusion_matrix,
    save_classification_report, FocalLoss
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        avg_loss, avg_f1
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
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    epoch_f1 = metrics['f1_score']
    
    return epoch_loss, epoch_f1


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        avg_loss, avg_f1, all_preds, all_labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    epoch_f1 = metrics['f1_score']
    
    return epoch_loss, epoch_f1, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, patience, output_dir):
    """
    Full training loop with early stopping and checkpointing.
    
    Returns:
        history dictionary with training metrics
    """
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best model based on F1-Score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  ✓ Best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"  Best F1: {best_f1:.4f} at epoch {best_epoch}")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation F1-Score: {best_f1:.4f} (Epoch {best_epoch})")
    print("="*70 + "\n")
    
    return history


def evaluate_on_test(model, test_loader, device, class_names, output_dir):
    """
    Evaluate best model on test set and generate reports.
    """
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute test metrics
    test_metrics = compute_metrics(all_labels, all_preds)
    
    print("\nTest Set Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Save confusion matrix
    save_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    
    # Save classification report
    save_classification_report(all_labels, all_preds, class_names, output_dir)
    
    print("="*70 + "\n")
    
    return test_metrics


def main(args):
    """
    Main execution function.
    """
    # Setup device with GPU selection (single or multi-GPU)
    use_multi_gpu = False
    gpu_ids = []
    
    if torch.cuda.is_available():
        if args.gpu_id == 'all':
            # Use all available GPUs
            gpu_ids = list(range(torch.cuda.device_count()))
            device = torch.device(f'cuda:{gpu_ids[0]}')
            use_multi_gpu = len(gpu_ids) > 1
        elif args.gpu_id is not None:
            # Parse comma-separated GPU IDs
            try:
                gpu_ids = [int(x.strip()) for x in args.gpu_id.split(',')]
                # Validate GPU IDs
                valid_gpu_ids = [gid for gid in gpu_ids if gid < torch.cuda.device_count()]
                if len(valid_gpu_ids) == 0:
                    print(f"⚠ Warning: No valid GPU IDs found. Available GPUs: {torch.cuda.device_count()}")
                    print(f"⚠ Falling back to GPU 0")
                    gpu_ids = [0]
                else:
                    gpu_ids = valid_gpu_ids
                    if len(gpu_ids) < len([int(x.strip()) for x in args.gpu_id.split(',')]):
                        print(f"⚠ Warning: Some GPU IDs are invalid. Using: {gpu_ids}")
                device = torch.device(f'cuda:{gpu_ids[0]}')
                use_multi_gpu = len(gpu_ids) > 1
            except ValueError:
                print(f"⚠ Warning: Invalid GPU ID format. Using GPU 0")
                gpu_ids = [0]
                device = torch.device('cuda:0')
        else:
            # Default to GPU 0
            gpu_ids = [0]
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"\n{'='*70}")
    print(f"DEVICE: {device}")
    if device.type == 'cuda':
        if use_multi_gpu:
            print(f"Multi-GPU Training: ENABLED")
            print(f"GPU IDs: {gpu_ids}")
            for gid in gpu_ids:
                print(f"  GPU {gid}: {torch.cuda.get_device_name(gid)} "
                      f"({torch.cuda.get_device_properties(gid).total_memory / 1024**3:.2f} GB)")
        else:
            print(f"Single-GPU Training")
            print(f"GPU ID: {gpu_ids[0]}")
            print(f"GPU Name: {torch.cuda.get_device_name(gpu_ids[0])}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_ids[0]).total_memory / 1024**3:.2f} GB")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    check_dir(output_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names, num_classes = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        seed=42
    )
    
    # Create model
    model = create_efficientnet_b0(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Wrap model with DataParallel for multi-GPU training
    if use_multi_gpu:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"✓ Model wrapped with DataParallel (GPUs: {gpu_ids})")
        print(f"✓ Effective batch size: {args.batch_size} × {len(gpu_ids)} = {args.batch_size * len(gpu_ids)}\n")
    
    # Get class weights
    class_weights = get_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)
    
    # Setup loss function
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print("✓ Using Focal Loss (gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("✓ Using Weighted Cross-Entropy Loss")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"✓ Optimizer: Adam (lr={args.lr}, weight_decay=1e-4)")
    
    # Setup scheduler (optional)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        print("✓ Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    else:
        scheduler = None
        print("✓ Scheduler: Disabled (constant learning rate)")
    print()
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir
    )
    
    # Save training curves
    save_loss_f1_curves(history, output_dir)
    
    # Load best model and evaluate on test set
    print("Loading best model for test evaluation...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['best_f1']:.4f})\n")
    
    # Evaluate on test set
    test_metrics = evaluate_on_test(model, test_loader, device, class_names, output_dir)
    
    # Save final summary
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("EFFICIENTNET-B0 TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: EfficientNet-B0 (Pretrained)\n")
        f.write(f"Input Resolution: 224x224\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Classes: {class_names}\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Device: {device}\n")
        if use_multi_gpu:
            f.write(f"  Multi-GPU: {gpu_ids}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  LR Scheduler: {'Enabled (ReduceLROnPlateau)' if args.use_scheduler else 'Disabled'}\n")
        f.write(f"  Loss Function: {args.loss_type}\n")
        f.write(f"  Early Stopping Patience: {args.patience}\n\n")
        f.write(f"Best Validation F1-Score: {checkpoint['best_f1']:.4f} (Epoch {checkpoint['epoch']})\n\n")
        f.write(f"Test Set Results:\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_metrics['f1_score']:.4f}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Training summary saved: {summary_path}")
    print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY! 🎉\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 for Eggplant Disease Detection')
    
    # Paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images",
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_pretrained",
        help='Path to save outputs'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (can be higher for 224x224)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--gpu_id', type=str, default=None, 
                       help='GPU ID(s) to use. Examples: "0" (single GPU), "0,1,2" (multi-GPU), "all" (all GPUs). Default: 0')
    parser.add_argument('--use_scheduler', action='store_true', 
                       help='Use ReduceLROnPlateau scheduler (default: disabled)')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'ce'], 
                       help='Loss function: focal or ce (cross-entropy)')
    
    args = parser.parse_args()
    
    main(args)
