"""
Helper Script: Inspect Checkpoint Structure
"""
import torch
from pathlib import Path

checkpoint_path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\efficientnet_feature_extraction\output_compressed\best_model.pth")

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f"\n{'='*60}")
print("CHECKPOINT STRUCTURE")
print(f"{'='*60}")

if isinstance(ckpt, dict):
    print(f"Type: Dictionary")
    print(f"Keys: {list(ckpt.keys())}")
    
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"\n{key}: Dictionary with {len(ckpt[key])} items")
            print(f"  Sample keys: {list(ckpt[key].keys())[:5]}")
        else:
            print(f"\n{key}: {type(ckpt[key])}")
else:
    print(f"Type: {type(ckpt)}")
    print(f"Direct state_dict with {len(ckpt)} keys")
    print(f"Sample keys: {list(ckpt.keys())[:10]}")
