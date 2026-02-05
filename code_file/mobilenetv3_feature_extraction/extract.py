"""
Feature Extraction Script for MobileNetV3 Small
===============================================
Main entry point for extracting 576-dimensional features from eggplant leaf images
using custom-trained MobileNetV3 weights.

Author: AI Engineer
Date: 2026-02-04

Usage:
    python extract.py
    python extract.py --data_dir "path/to/images" --output_dir "path/to/output"
    python extract.py --weight_path "path/to/custom_weights.pth"
"""

import argparse
import time
from pathlib import Path
from typing import List
import torch
import numpy as np

# Import custom modules
import config
from dataset import create_dataloader
from model import create_model
from utils import (
    ensure_directory,
    save_features_to_csv,
    print_extraction_summary,
    validate_csv_output
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract MobileNetV3 features from eggplant leaf images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(config.DATA_DIR),
        help='Path to directory containing images (supports subdirectories)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.OUTPUT_DIR),
        help='Path to directory for saving extracted features CSV'
    )
    
    parser.add_argument(
        '--weight_path',
        type=str,
        default=str(config.MODEL_WEIGHT_PATH),
        help='Path to custom trained MobileNetV3 weights (.pth file)'
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
        help='Number of data loading workers (use 0 for debugging on Windows)'
    )
    
    parser.add_argument(
        '--output_name',
        type=str,
        default=config.OUTPUT_CSV_NAME,
        help='Name of output CSV file'
    )
    
    return parser.parse_args()


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[np.ndarray, List[str], List[str]]:
    """
    Extract features from all images in dataloader.
    
    Args:
        model (torch.nn.Module): MobileNetV3 feature extractor
        dataloader (DataLoader): DataLoader containing images
        device (torch.device): Device to run inference on
    
    Returns:
        tuple: (features_array, filenames_list, labels_list)
            - features_array: NumPy array of shape [N, 576]
            - filenames_list: List of corresponding filenames
            - labels_list: List of corresponding class labels
    """
    model.eval()
    
    all_features = []
    all_filenames = []
    all_labels = []
    
    print("\n" + "="*70)
    print("STARTING FEATURE EXTRACTION")
    print("="*70)
    
    with torch.no_grad():  # Disable gradients for inference
        for batch_idx, (images, filenames, labels) in enumerate(dataloader, 1):
            # Move images to device
            images = images.to(device)  # [B, 3, 224, 224]
            
            # Extract features
            features = model(images)  # [B, 576]
            
            # Move to CPU and convert to NumPy
            features_np = features.cpu().numpy()  # [B, 576]
            
            # Accumulate results
            all_features.append(features_np)
            all_filenames.extend(filenames)
            all_labels.extend(labels)
            
            # Progress logging
            if batch_idx % 10 == 0 or batch_idx == len(dataloader):
                print(f"  Processed batch {batch_idx}/{len(dataloader)} "
                      f"({len(all_filenames)} images)")
    
    # Concatenate all batches
    features_array = np.vstack(all_features)  # [N, 576]
    
    print(f"\n✓ Feature extraction completed!")
    print(f"  - Total images: {features_array.shape[0]}")
    print(f"  - Feature dimension: {features_array.shape[1]}")
    print("="*70)
    
    return features_array, all_filenames, all_labels


def main() -> None:
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert string paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weight_path = Path(args.weight_path)
    output_csv_path = output_dir / args.output_name
    
    # Print configuration
    print("\n" + "="*70)
    print("MOBILENETV3 FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print(f"Data Directory    : {data_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Model Weights     : {weight_path}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Num Workers       : {args.num_workers}")
    print(f"Device            : {config.DEVICE}")
    print("="*70 + "\n")
    
    # Validate paths
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Create DataLoader
    print("Step 1: Creating DataLoader...")
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Step 2: Load Model
    print("\nStep 2: Loading MobileNetV3 Model...")
    model = create_model(weight_path=weight_path, device=config.DEVICE)
    
    # Step 3: Extract Features
    print("\nStep 3: Extracting Features...")
    features, filenames, labels = extract_features(
        model=model,
        dataloader=dataloader,
        device=config.DEVICE
    )
    
    # Step 4: Save to CSV
    print("\nStep 4: Saving Features to CSV...")
    save_features_to_csv(
        features=features,
        filenames=filenames,
        labels=labels,
        output_path=output_csv_path,
        feature_dim=config.FEATURE_DIM
    )
    
    # Step 5: Validate Output
    print("\nStep 5: Validating Output...")
    is_valid = validate_csv_output(
        csv_path=output_csv_path,
        expected_samples=len(filenames),
        expected_dim=config.FEATURE_DIM
    )
    
    if not is_valid:
        print("⚠ Warning: CSV validation failed. Please check the output file.")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print_extraction_summary(
        total_images=len(filenames),
        feature_dim=config.FEATURE_DIM,
        output_path=output_csv_path,
        elapsed_time=elapsed_time
    )
    
    print("✓ Feature extraction pipeline completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
