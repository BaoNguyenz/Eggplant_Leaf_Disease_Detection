"""
Example training commands for different scenarios.
Copy and paste these into PowerShell to run training.

Author: Senior AI Engineer
Date: 2026-02-05
"""

# =============================================================================
# SCENARIO 1: Quick Test (2 epochs, verify everything works)
# =============================================================================
"""
python train.py `
  --epochs 2 `
  --batch_size 16 `
  --lr 0.001 `
  --loss cross_entropy `
  --num_workers 0 `
  --output_dir "./test_outputs"
"""

# =============================================================================
# SCENARIO 2: Full Training with CrossEntropy Loss
# =============================================================================
"""
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss cross_entropy `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 10 `
  --num_workers 4 `
  --output_dir "./outputs_ce"
"""

# =============================================================================
# SCENARIO 3: Full Training with Focal Loss (Recommended for Imbalanced Data)
# =============================================================================
"""
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --epochs 100 `
  --batch_size 32 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss focal_loss `
  --focal_gamma 2.0 `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --num_workers 4 `
  --output_dir "./outputs_focal"
"""

# =============================================================================
# SCENARIO 4: Transfer Learning (Frozen Backbone, Fast Training)
# =============================================================================
"""
python train.py `
  --epochs 30 `
  --batch_size 64 `
  --lr 0.001 `
  --freeze_backbone `
  --loss cross_entropy `
  --num_workers 4 `
  --output_dir "./outputs_frozen"
"""

# =============================================================================
# SCENARIO 5: CPU-Only Training (No GPU)
# =============================================================================
"""
python train.py `
  --epochs 20 `
  --batch_size 8 `
  --lr 0.0001 `
  --device cpu `
  --num_workers 0 `
  --output_dir "./outputs_cpu"
"""

# =============================================================================
# SCENARIO 6: Hyperparameter Experimentation
# =============================================================================

# Experiment 1: High learning rate
"""
python train.py `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.0005 `
  --loss focal_loss `
  --output_dir "./exp_lr_high"
"""

# Experiment 2: Low learning rate
"""
python train.py `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.00001 `
  --loss focal_loss `
  --output_dir "./exp_lr_low"
"""

# Experiment 3: Large batch size
"""
python train.py `
  --epochs 50 `
  --batch_size 64 `
  --lr 0.0002 `
  --loss focal_loss `
  --output_dir "./exp_batch_large"
"""

# Experiment 4: Small batch size
"""
python train.py `
  --epochs 50 `
  --batch_size 16 `
  --lr 0.00005 `
  --loss focal_loss `
  --output_dir "./exp_batch_small"
"""

# =============================================================================
# RECOMMENDED PIPELINE FOR BEST RESULTS
# =============================================================================
"""
# Step 1: Quick validation (ensure no errors)
python train.py --epochs 2 --batch_size 16 --num_workers 0 --output_dir "./step1_validate"

# Step 2: Train frozen backbone (fast feature extraction)
python train.py `
  --epochs 20 `
  --batch_size 64 `
  --lr 0.001 `
  --freeze_backbone `
  --output_dir "./step2_frozen"

# Step 3: Full fine-tuning with best loss function
python train.py `
  --epochs 100 `
  --batch_size 32 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss focal_loss `
  --focal_gamma 2.0 `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --num_workers 4 `
  --output_dir "./step3_final"
"""
