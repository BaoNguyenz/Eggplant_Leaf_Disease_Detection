"""
Main training script for VGG16 Eggplant Leaf Disease Detection.
Uses Hugging Face Trainer with custom weighted loss for imbalanced data.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_class_weights, compute_metrics, check_dir, set_seed
from data_setup import create_dataloaders, EggplantDataset
from model_setup import load_model, save_model


# Default paths (Windows paths with raw strings)
DEFAULT_DATA_DIR = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images"
DEFAULT_OUTPUT_DIR = r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\VGG16_pretrained"


class HFDatasetWrapper(Dataset):
    """
    Wrapper to make PyTorch DataLoader compatible with HuggingFace Trainer.
    """
    
    def __init__(self, torch_dataset: EggplantDataset):
        self.torch_dataset = torch_dataset
        self.labels = torch_dataset.labels
        
    def __len__(self) -> int:
        return len(self.torch_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.torch_dataset[idx]
        return {
            'pixel_values': image,
            'labels': label
        }


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer with weighted CrossEntropyLoss for imbalanced datasets.
    """
    
    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use weighted CrossEntropyLoss.
        """
        labels = inputs.get('labels')
        pixel_values = inputs.get('pixel_values')
        
        # Forward pass - get SequenceClassifierOutput
        outputs = model(pixel_values, labels=labels)
        logits = outputs.logits
        
        # Compute weighted loss
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        
        # Return SequenceClassifierOutput with loss (supports subscripting)
        output_with_loss = SequenceClassifierOutput(loss=loss, logits=logits)
        
        return (loss, output_with_loss) if return_outputs else loss


class VGG16Wrapper(nn.Module):
    """
    Wrapper for VGG16 to make it compatible with HuggingFace Trainer.
    Returns SequenceClassifierOutput with logits attribute.
    """
    
    def __init__(self, vgg_model: nn.Module, num_labels: int = None):
        super().__init__()
        self.vgg = vgg_model
        # Create a config-like object for Trainer compatibility
        self.config = type('Config', (), {'num_labels': num_labels})()
        
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        logits = self.vgg(pixel_values)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Return SequenceClassifierOutput for HuggingFace Trainer compatibility
        return SequenceClassifierOutput(loss=loss, logits=logits)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train VGG16 for Eggplant Leaf Disease Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Path to save outputs (models, logs)'
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    
    # Model arguments
    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='Freeze VGG16 backbone layers'
    )
    
    # Early stopping arguments
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience (epochs)'
    )
    parser.add_argument(
        '--early_stopping_threshold',
        type=float,
        default=0.001,
        help='Early stopping threshold'
    )
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument(
        '--cuda_id',
        type=str,
        default='0',
        help='CUDA device ID(s). Use "-1" for CPU, "0" for GPU 0, "0,1" for GPU 0 and 1 (multi-GPU), etc.'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training (FP16)'
    )
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set CUDA device(s)
    cuda_id_str = args.cuda_id.strip()
    if cuda_id_str == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("[INFO] Using CPU (CUDA disabled)")
        num_gpus = 0
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id_str
        gpu_list = [int(x.strip()) for x in cuda_id_str.split(',')]
        num_gpus = len(gpu_list)
        if num_gpus == 1:
            print(f"[INFO] Using GPU {gpu_list[0]}")
        else:
            print(f"[INFO] Using {num_gpus} GPUs: {gpu_list}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check and create output directory
    output_dir = check_dir(args.output_dir)
    
    # Print configuration
    print("=" * 60)
    print("VGG16 Eggplant Leaf Disease Detection Training")
    print("=" * 60)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Image Size: {args.image_size}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    if torch.cuda.is_available() and num_gpus > 0:
        print(f"Device: CUDA ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
        for i in range(min(num_gpus, torch.cuda.device_count())):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Device: CPU")
    print("=" * 60)
    
    # Create dataloaders
    print("\n[STEP 1] Loading and splitting dataset...")
    train_loader, val_loader, test_loader, class_to_idx, class_names, train_dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_state=args.seed
    )
    
    num_classes = len(class_names)
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Calculate class weights
    print("\n[STEP 2] Calculating class weights for imbalanced data...")
    class_weights_dict = get_class_weights(train_dataset)
    class_weights = torch.FloatTensor([class_weights_dict[i] for i in range(num_classes)])
    print(f"[INFO] Class weights: {class_weights.tolist()}")
    
    # Load model
    print("\n[STEP 3] Loading VGG16-BN pretrained model...")
    vgg_model = load_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model = VGG16Wrapper(vgg_model, num_labels=num_classes)
    
    # Wrap datasets for HuggingFace Trainer
    train_hf_dataset = HFDatasetWrapper(train_dataset)
    val_hf_dataset = HFDatasetWrapper(val_loader.dataset)
    test_hf_dataset = HFDatasetWrapper(test_loader.dataset)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation strategy
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Saving
        save_total_limit=3,
        
        # Mixed precision
        fp16=args.fp16 and torch.cuda.is_available(),
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        report_to="none",
        
        # Other
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        
        # Disable default data removal behavior
        label_names=["labels"],
    )
    
    # Create callbacks
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    # Create trainer with weighted loss
    print("\n[STEP 4] Initializing Trainer with weighted loss...")
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=val_hf_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[early_stopping_callback],
    )
    
    # Start training
    print("\n[STEP 5] Starting training...")
    print("=" * 60)
    train_result = trainer.train()
    
    # Save training metrics
    print("\n[STEP 6] Training completed! Saving results...")
    trainer.save_model(str(output_dir / "best_model"))
    trainer.save_state()
    
    # Save the VGG model directly with class mappings
    save_model(
        model=model.vgg,
        save_path=str(output_dir / "vgg16_best.pth"),
        class_to_idx=class_to_idx
    )
    
    # Evaluate on test set
    print("\n[STEP 7] Evaluating on test set...")
    test_results = trainer.evaluate(test_hf_dataset)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Test Set)")
    print("=" * 60)
    print(f"  Accuracy:  {test_results.get('eval_accuracy', 0):.4f}")
    print(f"  Precision: {test_results.get('eval_precision', 0):.4f}")
    print(f"  Recall:    {test_results.get('eval_recall', 0):.4f}")
    print(f"  F1-Score:  {test_results.get('eval_f1', 0):.4f}")
    print("=" * 60)
    
    # Save results to file
    results_path = output_dir / "training_results.txt"
    with open(results_path, 'w') as f:
        f.write("VGG16 Eggplant Leaf Disease Detection - Training Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Data Directory: {args.data_dir}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Classes: {class_names}\n\n")
        f.write("Test Results:\n")
        f.write(f"  Accuracy:  {test_results.get('eval_accuracy', 0):.4f}\n")
        f.write(f"  Precision: {test_results.get('eval_precision', 0):.4f}\n")
        f.write(f"  Recall:    {test_results.get('eval_recall', 0):.4f}\n")
        f.write(f"  F1-Score:  {test_results.get('eval_f1', 0):.4f}\n")
    
    print(f"\n[INFO] Results saved to: {results_path}")
    print(f"[INFO] Best model saved to: {output_dir / 'best_model'}")
    print(f"[INFO] PyTorch model saved to: {output_dir / 'vgg16_best.pth'}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
