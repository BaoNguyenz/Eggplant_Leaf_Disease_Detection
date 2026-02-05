# Multi-GPU Training Guide for MobileNetV3

## 📋 Overview
This guide explains how to use multiple GPUs to train the MobileNetV3 Small model using PyTorch's `DataParallel`.

## 🚀 Quick Start

### Basic Multi-GPU Training (2 GPUs)
```bash
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64 --epochs 50
```

### Advanced Multi-GPU Training (4 GPUs with Focal Loss)
```bash
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --epochs 50 --loss_type focal --patience 10
```

## 🎛️ Multi-GPU Arguments

| Argument        | Type | Default  | Description                                    |
|-----------------|------|----------|------------------------------------------------|
| `--use_multi_gpu` | flag | False    | Enable multi-GPU training with DataParallel |
| `--gpu_ids`     | str  | "0,1,2,3" | Comma-separated GPU IDs to use              |

## 📊 Configuration Examples

### 1. Single GPU (Baseline)
```bash
python train.py --batch_size 32 --epochs 50
```
- **Batch size**: 32
- **GPUs**: 1
- **Use case**: Limited VRAM, testing, or only 1 GPU available

---

### 2. Dual GPU Setup
```bash
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64 --epochs 50
```
- **Batch size**: 64 (32 per GPU)
- **GPUs**: 2
- **Use case**: Recommended for most dual-GPU systems
- **Speedup**: ~1.7-1.9x compared to single GPU

---

### 3. Quad GPU Setup (Maximum Performance)
```bash
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --epochs 50 --loss_type focal
```
- **Batch size**: 128 (32 per GPU)
- **GPUs**: 4
- **Use case**: High-end workstations or servers
- **Speedup**: ~3.0-3.5x compared to single GPU

---

### 4. Custom GPU Selection
```bash
python train.py --use_multi_gpu --gpu_ids "0,2" --batch_size 64 --epochs 50
```
- **Batch size**: 64
- **GPUs**: Only GPU 0 and GPU 2 (skipping GPU 1 and 3)
- **Use case**: When specific GPUs are already in use or have different VRAM

---

### 5. High Batch Size Training (Large Memory)
```bash
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 256 --epochs 50 --lr 0.002
```
- **Batch size**: 256 (64 per GPU)
- **GPUs**: 4
- **Learning rate**: Increased to 0.002 (scaled with batch size)
- **Use case**: When you have GPUs with large VRAM (24GB+)

## ⚙️ How It Works

### 1. Model Wrapping
The code automatically wraps the model with `nn.DataParallel`:
```python
if args.use_multi_gpu:
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
    model = nn.DataParallel(model, device_ids=gpu_ids)
```

### 2. Data Distribution
- Each GPU receives a portion of the batch
- Forward pass runs in parallel on all GPUs
- Gradients are gathered and averaged across GPUs
- Backward pass updates the model on the primary GPU

### 3. Automatic Checkpoint Handling
The code handles DataParallel correctly when saving/loading:
```python
# Save: unwrap DataParallel
model_to_save = model.module if isinstance(model, nn.DataParallel) else model
torch.save({'model_state_dict': model_to_save.state_dict()}, path)

# Load: handle both wrapped and unwrapped models
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])
```

## 📈 Batch Size Recommendations

| GPUs | Recommended Batch Size | Per-GPU Batch Size | Total Memory |
|------|------------------------|--------------------| -------------|
| 1    | 32                     | 32                 | ~4-6 GB      |
| 2    | 64                     | 32                 | ~8-12 GB     |
| 4    | 128                    | 32                 | ~16-24 GB    |
| 4*   | 256                    | 64                 | ~32-48 GB    |

*Requires high-end GPUs (e.g., RTX 4090, A100)

## 🎯 Best Practices

### 1. Scale Batch Size with GPU Count
```bash
# 1 GPU
python train.py --batch_size 32

# 2 GPUs → 2x batch size
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64

# 4 GPUs → 4x batch size
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128
```

### 2. Adjust Learning Rate for Large Batches
When using large batch sizes (128+), consider increasing learning rate:
```bash
# Small batch (32)
python train.py --batch_size 32 --lr 0.001

# Large batch (128) → scale lr by sqrt(batch_ratio)
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --lr 0.002
```

Rule of thumb: `new_lr = base_lr * sqrt(new_batch / base_batch)`

### 3. Monitor GPU Utilization
Use `nvidia-smi` to check GPU usage:
```bash
# Windows
nvidia-smi

# Watch mode (update every 1 second)
nvidia-smi -l 1
```

### 4. Handle OOM (Out of Memory) Errors
If you encounter CUDA out of memory errors:
```bash
# Reduce batch size
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 48  # instead of 64

# Or reduce number of GPUs
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64  # instead of "0,1,2,3"
```

## 🔍 Checking Your GPU Setup

### View Available GPUs
```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Expected Output
```
[INFO] Multi-GPU training enabled with 4 GPUs
[INFO]   GPU 0: NVIDIA GeForce RTX 4090
[INFO]   GPU 1: NVIDIA GeForce RTX 4090
[INFO]   GPU 2: NVIDIA GeForce RTX 4090
[INFO]   GPU 3: NVIDIA GeForce RTX 4090
[INFO] Model wrapped with DataParallel (GPUs: [0, 1, 2, 3])
```

## ⚠️ Important Notes

### 1. DataParallel vs DistributedDataParallel
- This implementation uses **DataParallel** (easier to use, single-process)
- For multi-node training, consider **DistributedDataParallel** (more complex, better scaling)

### 2. Linear Scaling Limitation
- Speedup is **not** perfectly linear (2 GPUs ≠ 2x speed)
- Expected speedup: 1.7-1.9x for 2 GPUs, 3.0-3.5x for 4 GPUs
- Communication overhead between GPUs reduces efficiency

### 3. Reproducibility with Multi-GPU
Multi-GPU training may produce slightly different results even with fixed seeds due to:
- Non-deterministic GPU operations
- Different batch splitting across GPUs

For maximum reproducibility, use single GPU with `--num_workers 0`

## 🎬 Complete Example Workflows

### Workflow 1: Quick Test (2 GPUs, 5 epochs)
```bash
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_pretrained"
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64 --epochs 5 --patience 3
```

### Workflow 2: Full Training (4 GPUs, Focal Loss)
```bash
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_pretrained"
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --epochs 50 --loss_type focal --patience 10 --lr 0.002
```

### Workflow 3: Conservative Training (2 GPUs, Cross Entropy)
```bash
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\mobilenetv3_pretrained"
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 64 --epochs 50 --loss_type cross_entropy --patience 10 --lr 0.001
```

## 🛠️ Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size or number of GPUs
```bash
python train.py --use_multi_gpu --gpu_ids "0,1" --batch_size 48
```

### Problem: "only 1 GPU available but 4 requested"
**Solution**: Adjust `--gpu_ids` to match available GPUs
```bash
python train.py --use_multi_gpu --gpu_ids "0" --batch_size 32
```

### Problem: Slow training with multi-GPU
**Solution**: Check if `num_workers` is sufficient
```bash
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --num_workers 8
```
Recommended: `num_workers = 2 * num_gpus`

## ✅ Summary

**To enable multi-GPU training:**
1. Add `--use_multi_gpu` flag
2. Specify GPU IDs with `--gpu_ids "0,1,..."` 
3. Scale batch size accordingly
4. Optionally adjust learning rate for large batches

**Example command:**
```bash
python train.py --use_multi_gpu --gpu_ids "0,1,2,3" --batch_size 128 --epochs 50 --loss_type focal
```

That's it! The code handles all the complexity of DataParallel automatically. 🚀
