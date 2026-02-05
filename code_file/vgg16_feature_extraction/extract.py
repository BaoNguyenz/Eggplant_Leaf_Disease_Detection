"""
Main Feature Extraction Script
Extracts features from images using VGG16 model (with custom weights support).
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Import local modules
from config import Config
from model import create_feature_extractor
from dataset import create_dataloader
from utils import (
    save_features_to_csv,
    save_metadata,
    print_extraction_summary,
    create_output_filename,
    normalize_path_for_windows
)


def extract_features(model, dataloader, device: str) -> tuple:
    """
    Extract features from all images in dataloader.
    
    Args:
        model: VGG16 feature extractor model
        dataloader: DataLoader containing images
        device: Device to run inference on
        
    Returns:
        tuple: (features, labels, image_paths)
    """
    all_features = []
    all_labels = []
    all_paths = []
    
    print("\n🔄 Extracting features...")
    
    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in tqdm(dataloader, desc="Processing batches"):
            # Move images to device
            batch_images = batch_images.to(device)
            
            # Extract features
            features = model(batch_images)
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy()
            
            # Accumulate results
            all_features.append(features_np)
            all_labels.extend(batch_labels)
            all_paths.extend(batch_paths)
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    
    print(f"✓ Extracted features shape: {all_features.shape}")
    
    return all_features, all_labels, all_paths


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='VGG16 Feature Extraction with Custom Pretrained Weights Support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(Config.DEFAULT_DATA_DIR),
        help='Path to dataset directory containing class subdirectories'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(Config.DEFAULT_OUTPUT_DIR),
        help='Path to output directory for saving features'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=Config.BATCH_SIZE,
        help='Batch size for feature extraction'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=Config.NUM_WORKERS,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--output_filename',
        type=str,
        default=None,
        help='Custom output filename (if not provided, timestamped name will be used)'
    )
    
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    
    parser.add_argument(
        '--custom_weights',
        type=str,
        default=None,
        help='Path to custom pretrained weights (.pth file). If not specified, uses Config.CUSTOM_WEIGHTS_PATH'
    )
    
    parser.add_argument(
        '--use_imagenet',
        action='store_true',
        help='Force using ImageNet weights instead of custom weights'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Print header
    print("\n" + "="*60)
    print("VGG16 FEATURE EXTRACTION")
    print("With Custom Fine-tuned Weights Support")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Convert paths to pathlib.Path objects and validate
        data_dir = normalize_path_for_windows(Path(args.data_dir))
        output_dir = normalize_path_for_windows(Path(args.output_dir))
        
        data_dir, output_dir = Config.validate_paths(data_dir, output_dir)
        
        print(f"\n📁 Configuration:")
        print(f"  - Data directory:   {data_dir}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Batch size:       {args.batch_size}")
        print(f"  - Num workers:      {args.num_workers}")
        
        # Determine device
        if args.no_gpu:
            device = 'cpu'
            print("\n⚠ GPU disabled by user (--no_gpu flag)")
        else:
            if torch.cuda.is_available():
                if args.gpu_id >= torch.cuda.device_count():
                    print(f"\n⚠ Warning: GPU {args.gpu_id} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
                    print(f"  Using GPU 0 instead.")
                    device = 'cuda:0'
                else:
                    device = f'cuda:{args.gpu_id}'
                    print(f"\n✓ Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            else:
                device = 'cpu'
                print("\n⚠ GPU not available, using CPU")
        
        # Determine which weights to use
        # Priority: --use_imagenet flag > --custom_weights arg > Config.CUSTOM_WEIGHTS_PATH > ImageNet
        custom_weights_path = None
        
        print("\n" + "="*60)
        if args.use_imagenet:
            print("📦 WEIGHTS SOURCE: ImageNet Pretrained")
            print("   (Forced by --use_imagenet flag)")
        elif args.custom_weights:
            custom_weights_path = Path(args.custom_weights)
            if not custom_weights_path.exists():
                raise FileNotFoundError(f"Custom weights file not found: {custom_weights_path}")
            print("📦 WEIGHTS SOURCE: Custom Fine-tuned Weights")
            print(f"   (From CLI argument)")
        elif Config.CUSTOM_WEIGHTS_PATH and Config.CUSTOM_WEIGHTS_PATH.exists():
            custom_weights_path = Config.CUSTOM_WEIGHTS_PATH
            print("📦 WEIGHTS SOURCE: Custom Fine-tuned Weights")
            print(f"   (From config.CUSTOM_WEIGHTS_PATH)")
        else:
            print("📦 WEIGHTS SOURCE: ImageNet Pretrained")
            print("   (Default fallback)")
        print("="*60)
        
        # Create model
        print("\n🔧 Loading VGG16 Model...")
        model = create_feature_extractor(
            pretrained=Config.USE_PRETRAINED, 
            device=device,
            custom_weights_path=str(custom_weights_path) if custom_weights_path else None
        )
        
        # Create dataloader
        print("\n📊 Loading Dataset...")
        dataloader, dataset = create_dataloader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            img_size=Config.IMG_SIZE,
            mean=Config.MEAN,
            std=Config.STD,
            num_workers=args.num_workers,
            shuffle=False
        )
        
        # Start extraction
        start_time = time.time()
        
        features, labels, image_paths = extract_features(model, dataloader, device)
        
        extraction_time = time.time() - start_time
        
        # Determine output filename
        if args.output_filename:
            output_filename = args.output_filename
            if not output_filename.endswith('.csv'):
                output_filename += '.csv'
        else:
            prefix = 'vgg16_custom_features' if custom_weights_path else 'vgg16_imagenet_features'
            output_filename = create_output_filename(prefix, 'csv')
        
        output_path = output_dir / output_filename
        
        # Save features to CSV
        print("\n💾 Saving Features to CSV...")
        save_features_to_csv(
            features=features,
            labels=labels,
            image_paths=image_paths,
            output_path=output_path,
            encoding=Config.OUTPUT_ENCODING
        )
        
        # Save metadata
        config_dict = {
            'model_name': Config.MODEL_NAME,
            'feature_dim': Config.FEATURE_DIM,
            'img_size': list(Config.IMG_SIZE),
            'batch_size': args.batch_size,
            'weights_source': 'custom' if custom_weights_path else 'imagenet',
            'custom_weights_path': str(custom_weights_path) if custom_weights_path else None
        }
        
        save_metadata(
            output_dir=output_dir,
            config=config_dict,
            num_samples=len(features),
            num_classes=len(dataset.class_names),
            class_names=dataset.class_names,
            extraction_time=extraction_time
        )
        
        # Print summary
        print_extraction_summary(
            num_samples=len(features),
            num_classes=len(dataset.class_names),
            class_names=dataset.class_names,
            feature_dim=Config.FEATURE_DIM,
            extraction_time=extraction_time
        )
        
        print("\n✅ Feature extraction completed successfully!")
        print(f"📁 Output file: {output_path}\n")
        return 0
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
