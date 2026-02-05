"""
Main Extraction Script for DenseNet121 Feature Extraction
==========================================================
Entry point for extracting features from images using DenseNet121 with custom trained weights.

Usage:
    python extract.py
    python extract.py --data_dir "path/to/data" --output_dir "path/to/output"
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ImageDataset, get_transform
from model import create_feature_extractor
from utils import (
    check_and_create_dir,
    validate_data_dir,
    save_features_to_csv,
    print_extraction_summary
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DenseNet121 Feature Extraction for Eggplant Leaf Disease Detection (Custom Weights)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(config.DEFAULT_DATA_DIR),
        help='Path to directory containing class subdirectories with images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.DEFAULT_OUTPUT_DIR),
        help='Path to directory where output CSV will be saved'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size for feature extraction'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=config.NUM_WORKERS,
        help='Number of workers for data loading'
    )
    
    return parser.parse_args()


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract features from all images in the dataloader.
    
    Args:
        model: Feature extractor model
        dataloader: DataLoader with images
        device: Device to run inference on
        
    Returns:
        Tuple of (features, labels, filenames)
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\n{'='*70}")
    print(f"Extracting features...")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Processing batches"):
            # Move images to device
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            
            # Accumulate results
            all_features.append(features)
            all_labels.extend(labels)
            all_filenames.extend(filenames)
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    
    print(f"✓ Feature extraction complete!")
    print(f"  Shape: {all_features.shape}")
    
    return all_features, all_labels, all_filenames


def main() -> None:
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"DENSENET121 FEATURE EXTRACTION (CUSTOM WEIGHTS)")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*70}\n")
    
    # Step 1: Validate data directory
    try:
        validate_data_dir(data_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 2: Create output directory
    try:
        check_and_create_dir(output_dir)
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        return
    
    # Step 3: Create dataset and dataloader
    try:
        print(f"\n{'='*70}")
        print(f"Loading dataset...")
        print(f"{'='*70}")
        
        transform = get_transform()
        dataset = ImageDataset(data_dir, transform=transform)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Keep order for reproducibility
            num_workers=args.num_workers,
            pin_memory=config.PIN_MEMORY
        )
        
        print(f"✓ DataLoader created")
        print(f"  Total batches: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Step 4: Create feature extractor
    try:
        print(f"\n{'='*70}")
        print(f"Initializing model...")
        print(f"{'='*70}")
        
        model, device = create_feature_extractor()
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Step 5: Extract features
    try:
        features, labels, filenames = extract_features(model, dataloader, device)
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return
    
    # Step 6: Save features to CSV
    try:
        output_csv = output_dir / config.OUTPUT_CSV_NAME
        save_features_to_csv(features, labels, filenames, output_csv)
        
    except Exception as e:
        print(f"❌ Error saving features: {e}")
        return
    
    # Step 7: Print summary
    print_extraction_summary(
        num_images=len(dataset),
        num_classes=len(dataset.class_names),
        feature_dim=model.get_feature_dim(),
        device=str(device),
        output_path=output_csv
    )
    
    print("✅ Feature extraction completed successfully!\n")


if __name__ == "__main__":
    main()
