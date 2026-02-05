"""
Main script for ResNet34 feature extraction
Entry point for the feature extraction pipeline
"""
import argparse
import time
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom modules
import config
from dataset import ImageDataset, get_transform
from model import ResNet34FeatureExtractor
from utils import (
    ensure_directory,
    validate_directory,
    validate_file,
    save_features_to_csv,
    print_extraction_summary
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ResNet34 Feature Extraction for Eggplant Leaf Disease Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
    # Use default paths from config
    python extract.py
    
    # Specify custom paths
    python extract.py --data_dir "E:/Custom/Data/Path" --output_dir "E:/Custom/Output"
    
    # Use different model weights
    python extract.py --weight_path "E:/Custom/Weights/model.pth"
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(config.DATA_DIR),
        help=f'Directory containing classified images (default: {config.DATA_DIR})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.OUTPUT_DIR),
        help=f'Directory to save extracted features (default: {config.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--weight_path',
        type=str,
        default=str(config.MODEL_WEIGHT_PATH),
        help=f'Path to custom .pth weights file (default: {config.MODEL_WEIGHT_PATH})'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help=f'Batch size for processing (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=config.NUM_WORKERS,
        help=f'Number of DataLoader workers (default: {config.NUM_WORKERS})'
    )
    
    parser.add_argument(
        '--output_name',
        type=str,
        default=config.OUTPUT_CSV_NAME,
        help=f'Output CSV filename (default: {config.OUTPUT_CSV_NAME})'
    )
    
    return parser.parse_args()


def extract_features(
    model: ResNet34FeatureExtractor,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Extract features from all images in dataloader
    
    Args:
        model: Feature extraction model
        dataloader: DataLoader containing images
        device: Device to run inference on
        
    Returns:
        Tuple of (features_list, labels_list, filenames_list)
    """
    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    filenames_list: List[str] = []
    
    print("\nExtracting features...")
    model.eval()
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Processing batches"):
            # Move images to device
            images = images.to(device)
            
            # Extract features
            features = model.extract_features(images)
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy()
            
            # Store results
            features_list.extend(features_np)
            labels_list.extend(labels)
            filenames_list.extend(filenames)
    
    return features_list, labels_list, filenames_list


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert paths to pathlib.Path
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weight_path = Path(args.weight_path)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("ResNet34 Feature Extraction Pipeline")
    print("=" * 80)
    print(f"Data Directory     : {data_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Model Weights      : {weight_path}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Number of Workers  : {args.num_workers}")
    print(f"Device             : {config.DEVICE}")
    print("=" * 80 + "\n")
    
    # Validate paths
    try:
        validate_directory(data_dir, "Data directory")
        validate_file(weight_path, "Model weights file")
        ensure_directory(output_dir)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Create dataset and dataloader
    print("Step 1: Loading dataset...")
    transform = get_transform(
        config.IMAGE_SIZE, 
        config.IMAGENET_MEAN, 
        config.IMAGENET_STD
    )
    
    dataset = ImageDataset(data_dir, transform=transform)
    print(f"  └─ Total images: {len(dataset)}")
    print(f"  └─ Class distribution: {dataset.get_class_distribution()}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    print(f"  └─ Number of batches: {len(dataloader)}")
    
    # Step 2: Initialize model
    print("\nStep 2: Initializing model...")
    model = ResNet34FeatureExtractor(
        weight_path=weight_path,
        device=str(config.DEVICE)
    )
    
    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
    features, labels, filenames = extract_features(model, dataloader, config.DEVICE)
    print(f"  └─ Extracted {len(features)} feature vectors")
    
    # Step 4: Save to CSV
    print("\nStep 4: Saving features to CSV...")
    output_path = output_dir / args.output_name
    save_features_to_csv(
        features=features,
        labels=labels,
        filenames=filenames,
        output_path=output_path,
        encoding=config.CSV_ENCODING
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print_extraction_summary(
        total_images=len(features),
        feature_dim=config.FEATURE_DIM,
        output_path=output_path,
        elapsed_time=elapsed_time
    )
    
    print("\n✅ Feature extraction completed successfully!\n")


if __name__ == "__main__":
    main()
