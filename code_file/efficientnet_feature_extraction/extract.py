"""
Main Extraction Script for EfficientNet-B0 Feature Extraction
=============================================================
Entry point với argparse CLI support.
Trích xuất đặc trưng từ dataset và lưu vào CSV.
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import sys

# Import các module tự viết
from config import Config
from dataset import (
    EggplantLeafDataset,
    get_efficientnet_b0_transforms,
    create_dataloader
)
from model import (
    load_feature_extractor,
    verify_output_shape
)
from utils import (
    ensure_dir,
    get_image_paths,
    save_features_to_csv,
    validate_features,
    print_device_info
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 Feature Extraction for Eggplant Leaf Disease Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images",
        help="Đường dẫn tới thư mục Classified Images chứa các class folders"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction",
        help="Đường dẫn thư mục lưu file CSV kết quả"
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size cho feature extraction (thấp do model nặng, GPU limited)"
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help="Số worker processes cho DataLoader"
    )
    
    parser.add_argument(
        '--output_filename',
        type=str,
        default="efficientnet_b0_features.csv",
        help="Tên file CSV output"
    )
    
    parser.add_argument(
        '--custom_weights',
        type=str,
        default=None,
        help="Đường dẫn tới custom trained weights (.pth file). Nếu không chỉ định, sử dụng ImageNet pretrained weights"
    )
    
    return parser.parse_args()


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Trích xuất features cho toàn bộ dataset.
    
    Args:
        model: Feature extractor model
        dataloader: DataLoader chứa dataset
        device: Device để chạy model
    
    Returns:
        Tuple (all_features, all_image_paths)
        
    Note:
        img_paths từ DataLoader là strings (converted từ Path trong Dataset)
    """
    model.eval()  # Set to evaluation mode
    
    all_features = []
    all_image_paths = []
    
    print("\n========================================")
    print("Starting Feature Extraction...")
    print("========================================")
    
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (images, img_paths) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            # Move images to device
            images = images.to(device)
            
            # Forward pass
            features = model(images)  # [batch_size, 1280]
            
            # Collect results
            all_features.append(features.cpu())  # Move to CPU để tiết kiệm GPU memory
            # img_paths là list of strings từ DataLoader
            all_image_paths.extend(img_paths)
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)  # [total_images, 1280]
    
    print(f"\n✓ Feature extraction completed!")
    print(f"  - Total images processed: {len(all_image_paths)}")
    print(f"  - Features shape: {all_features.shape}")
    
    return all_features, all_image_paths


def main():
    """Main execution function."""
    
    # 1. Parse arguments
    args = parse_arguments()
    
    # 2. Update và validate config
    Config.update_paths(data_dir=args.data_dir, output_dir=args.output_dir)
    Config.update_batch_size(batch_size=args.batch_size)
    Config.update_custom_weights(custom_weights=args.custom_weights)
    Config.NUM_WORKERS = args.num_workers
    
    try:
        Config.validate_paths()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # 3. Print configuration
    print(Config.get_info())
    print_device_info()
    
    # 4. Ensure output directory exists
    ensure_dir(Config.OUTPUT_DIR)
    
    # 5. Get image paths
    print("\n========================================")
    print("Loading Dataset...")
    print("========================================")
    image_paths = get_image_paths(Config.DATA_DIR)
    
    if len(image_paths) == 0:
        print("❌ No images found! Please check your data directory.")
        sys.exit(1)
    
    # 6. Create transforms
    transform = get_efficientnet_b0_transforms(
        img_size=Config.IMG_SIZE,
        mean=Config.IMAGENET_MEAN,
        std=Config.IMAGENET_STD
    )
    print(f"✓ Transforms created for image size: {Config.IMG_SIZE}")
    
    # 7. Create dataset and dataloader
    dataset = EggplantLeafDataset(image_paths=image_paths, transform=transform)
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=False  # Không shuffle để đảm bảo thứ tự
    )
    print(f"✓ DataLoader created: {len(dataloader)} batches")
    
    # 8. Load model
    print("\n" + "="*50)
    print("Loading EfficientNet-B0 Model...")
    print("="*50)
    
    # Determine whether to use custom weights or ImageNet pretrained
    use_pretrained = Config.CUSTOM_WEIGHTS_PATH is None
    model = load_feature_extractor(
        device=Config.DEVICE, 
        pretrained=use_pretrained,
        custom_weights_path=Config.CUSTOM_WEIGHTS_PATH
    )
    
    # 9. Verify model output shape
    verify_output_shape(model, input_size=Config.IMG_SIZE)
    
    # 10. Extract features
    all_features, all_image_paths = extract_features(
        model=model,
        dataloader=dataloader,
        device=Config.DEVICE
    )
    
    # 11. Validate features
    try:
        validate_features(all_features, expected_dim=Config.FEATURE_DIM)
    except ValueError as e:
        print(f"❌ Feature validation failed: {e}")
        sys.exit(1)
    
    # 12. Save to CSV
    print("\n========================================")
    print("Saving Features to CSV...")
    print("========================================")
    output_csv_path = Config.OUTPUT_DIR / args.output_filename
    
    save_features_to_csv(
        features=all_features,
        image_paths=all_image_paths,
        data_dir=Config.DATA_DIR,
        output_path=output_csv_path,
        feature_dim=Config.FEATURE_DIM
    )
    
    # 13. Final summary
    print("\n========================================")
    print("✓ FEATURE EXTRACTION COMPLETED!")
    print("========================================")
    print(f"Total images   : {len(all_image_paths)}")
    print(f"Feature shape  : {all_features.shape}")
    print(f"Output CSV     : {output_csv_path}")
    print(f"File size      : {output_csv_path.stat().st_size / (1024*1024):.2f} MB")
    print("========================================\n")


if __name__ == "__main__":
    main()
