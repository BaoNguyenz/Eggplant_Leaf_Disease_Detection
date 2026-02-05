"""
Example Usage Script for MobileNetV3 Feature Extraction
=======================================================
This script demonstrates different ways to use the feature extraction system.

Author: AI Engineer
Date: 2026-02-04
"""

import subprocess
import sys
from pathlib import Path


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70 + "\n")


def run_command(cmd: list, description: str) -> None:
    """
    Run a command and display output.
    
    Args:
        cmd (list): Command to run as list of strings
        description (str): Description of what the command does
    """
    print(f"📌 {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        print(f"✓ Exit code: {result.returncode}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    
    print_section("MOBILENETV3 FEATURE EXTRACTION - USAGE EXAMPLES")
    
    # Example 1: Test Configuration
    print_section("Example 1: Test Configuration Module")
    run_command(
        [sys.executable, "config.py"],
        "Validate paths and print configuration"
    )
    
    # Example 2: Test Dataset
    print_section("Example 2: Test Dataset Module")
    run_command(
        [sys.executable, "dataset.py"],
        "Test image loading and preprocessing"
    )
    
    # Example 3: Test Model (if weights exist)
    print_section("Example 3: Test Model Module")
    weight_path = Path(r"E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_feature_extraction\mobilenetv3_output\best_model.pth")
    
    if weight_path.exists():
        run_command(
            [sys.executable, "model.py"],
            "Test model loading and forward pass"
        )
    else:
        print(f"⚠ Model weights not found at: {weight_path}")
        print("  Skipping model test. Please train the model first.\n")
    
    # Example 4: Test Utilities
    print_section("Example 4: Test Utility Functions")
    run_command(
        [sys.executable, "utils.py"],
        "Test CSV saving/loading and validation"
    )
    
    # Example 5: Full Feature Extraction (commented out - uncomment to run)
    print_section("Example 5: Full Feature Extraction (Ready to Run)")
    print("📌 To run full feature extraction, execute:")
    print("\n  python extract.py\n")
    print("Or with custom parameters:")
    print('  python extract.py --batch_size 64 --num_workers 4\n')
    
    # Uncomment below to run automatically:
    # run_command(
    #     [sys.executable, "extract.py", "--batch_size", "32"],
    #     "Extract features with batch_size=32"
    # )
    
    print_section("EXAMPLES COMPLETED")
    print("✓ All individual module tests completed!")
    print("\n💡 Next Steps:")
    print("  1. Run: python extract.py")
    print("  2. Check output: extracted_features/mobilenetv3_features.csv")
    print("  3. Use features for classification/clustering\n")
