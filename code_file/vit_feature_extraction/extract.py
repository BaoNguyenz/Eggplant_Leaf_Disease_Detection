"""
Main script for ViT feature extraction.
Extracts 768-dimensional features from eggplant leaf disease images.

Usage:
    python extract.py --data_dir "path/to/images" --output_dir "path/to/output"
    
Example:
    python extract.py
"""
import argparse
from pathlib import Path
from typing import List, Tuple
import time

import torch
import numpy as np
from tqdm import tqdm

from config import config
from dataset import create_dataloader
from model import create_feature_extractor
from utils import (
    get_device,
    create_output_dir,
    save_features_to_csv,
    print_extraction_summary,
    validate_paths
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract ViT features from eggplant leaf disease images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(config.DATA_DIR),
        help="Path to directory containing classified images"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Path to output directory for saving features"
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for feature extraction"
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=config.OUTPUT_CSV,
        help="Output CSV filename"
    )
    
    parser.add_argument(
        '--custom_weights',
        type=str,
        default=str(config.CUSTOM_WEIGHTS_PATH) if config.USE_CUSTOM_WEIGHTS else None,
        help="Path to custom trained model weights (.pth file). If not provided, uses ImageNet pretrained weights."
    )
    
    parser.add_argument(
        '--use_imagenet',
        action='store_true',
        help="Force use of ImageNet pretrained weights instead of custom weights"
    )
    
    return parser.parse_args()


def extract_features_from_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Extract features from a batch of images.
    
    Args:
        model: ViT feature extractor model
        images: Batch of images [batch_size, 3, 224, 224]
        device: Computing device
        
    Returns:
        Feature array of shape [batch_size, 768]
    """
    # Move images to device
    images = images.to(device)
    
    # Extract features (no gradient required)
    with torch.no_grad():
        features = model.extract_features(images)
    
    # Move back to CPU and convert to numpy
    features = features.cpu().numpy()
    
    return features


def run_extraction(
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    output_csv: str,
    custom_weights_path: str = None
) -> None:
    """
    Run complete feature extraction pipeline.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory
        batch_size: Batch size for extraction
        output_csv: Output CSV filename
        custom_weights_path: Optional path to custom trained weights
    """
    print("\n" + "="*70)
    print(" " * 15 + "ViT FEATURE EXTRACTION PIPELINE")
    print("="*70 + "\n")
    
    # Validate paths
    validate_paths(data_dir, output_dir)
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Setup device
    device = get_device(use_cuda=config.USE_CUDA)
    
    # Create dataloader
    print("\n[1/4] Loading dataset...")
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    print("\n[2/4] Loading ViT model...")
    model = create_feature_extractor(
        device=device,
        custom_weights_path=custom_weights_path,
        use_pretrained=True
    )
    
    # Extract features
    print("\n[3/4] Extracting features...")
    all_features: List[np.ndarray] = []
    all_paths: List[str] = []
    all_labels: List[str] = []
    
    start_time = time.time()
    
    with tqdm(total=len(dataloader), desc="Processing batches") as pbar:
        for batch_images, batch_paths, batch_labels in dataloader:
            # Extract features from batch
            batch_features = extract_features_from_batch(
                model=model,
                images=batch_images,
                device=device
            )
            
            # Store results
            all_features.append(batch_features)
            all_paths.extend(batch_paths)
            all_labels.extend(batch_labels)
            
            pbar.update(1)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Extraction completed in {elapsed_time:.2f} seconds")
    print(f"[INFO] Processing speed: {len(all_paths) / elapsed_time:.2f} images/second")
    
    # Save to CSV
    print("\n[4/4] Saving features to CSV...")
    output_csv_path = output_dir / output_csv
    save_features_to_csv(
        features=all_features,
        image_paths=all_paths,
        labels=all_labels,
        output_path=output_csv_path,
        feature_dim=config.FEATURE_DIM
    )
    
    # Print summary
    num_classes = len(set(all_labels))
    print_extraction_summary(
        num_images=len(all_paths),
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        output_csv=output_csv_path,
        device=device
    )


def main():
    """Main entry point for feature extraction."""
    # Parse arguments
    args = parse_arguments()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    try:
        # Determine custom weights path
        custom_weights_path = None
        if not args.use_imagenet:
            if args.custom_weights and args.custom_weights.lower() != 'none':
                custom_weights_path = args.custom_weights
                print(f"[INFO] Will use custom weights: {custom_weights_path}\n")
            elif config.USE_CUSTOM_WEIGHTS and config.CUSTOM_WEIGHTS_PATH:
                custom_weights_path = str(config.CUSTOM_WEIGHTS_PATH)
                print(f"[INFO] Will use custom weights from config: {custom_weights_path}\n")
        
        if not custom_weights_path:
            print("[INFO] Will use ImageNet pretrained weights\n")
        
        # Run extraction
        run_extraction(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            output_csv=args.output_csv,
            custom_weights_path=custom_weights_path
        )
        
        print("[SUCCESS] Feature extraction completed successfully! ✓\n")
        
    except Exception as e:
        print(f"\n[ERROR] Feature extraction failed: {e}\n")
        raise


if __name__ == "__main__":
    main()
