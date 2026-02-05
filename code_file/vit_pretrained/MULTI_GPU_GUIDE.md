# Hướng Dẫn Multi-GPU Training

Hệ thống đã được cập nhật để hỗ trợ **Multi-GPU Training** sử dụng **DataParallel**.

## 🎯 Lệnh Training với 2 GPUs (GPU 0 và GPU 1)

### Lệnh Đầy Đủ với Focal Loss (Khuyến nghị)

```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --gpu_ids "0,1" `
  --epochs 100 `
  --batch_size 64 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss focal_loss `
  --focal_gamma 2.0 `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --num_workers 8 `
  --seed 42 `
  --output_dir "./outputs_dual_gpu"
```

### Lệnh với CrossEntropy Loss

```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --gpu_ids "0,1" `
  --epochs 100 `
  --batch_size 64 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss cross_entropy `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --num_workers 8 `
  --seed 42 `
  --output_dir "./outputs_dual_gpu_ce"
```

---

## 📊 So Sánh Single GPU vs Multi-GPU

| Cấu hình | Batch Size | Num Workers | Thời gian/epoch | Ghi chú |
|----------|------------|-------------|-----------------|---------|
| **Single GPU (GPU 0)** | 32 | 4 | ~X phút | Baseline |
| **Multi-GPU (GPU 0,1)** | 64 | 8 | ~X/2 phút | Gấp đôi batch size |

---

## ⚙️ Tùy chọn GPU IDs

### Chỉ sử dụng GPU 0 (Single GPU)
```powershell
--gpu_ids "0"
# hoặc bỏ qua tham số này (default)
```

### Sử dụng GPU 0 và GPU 1 (Multi-GPU)
```powershell
--gpu_ids "0,1"
```

### Chỉ sử dụng GPU 1
```powershell
--gpu_ids "1"
```

### Sử dụng cả 3 GPUs (nếu có)
```powershell
--gpu_ids "0,1,2"
```

---

## 💡 Tips cho Multi-GPU Training

### 1. **Tăng Batch Size**
Với 2 GPUs, bạn có thể tăng batch size gấp đôi:
```powershell
# Single GPU: batch_size=32
--batch_size 32 --gpu_ids "0"

# Multi-GPU: batch_size=64 (32 per GPU)
--batch_size 64 --gpu_ids "0,1"
```

### 2. **Tăng Num Workers**
Với nhiều GPU, tăng số workers để load data nhanh hơn:
```powershell
# Single GPU
--num_workers 4

# Multi-GPU
--num_workers 8  # Hoặc thậm chí 16
```

### 3. **Điều chỉnh Learning Rate**
Khi batch size tăng, có thể cần điều chỉnh learning rate:
```powershell
# Single GPU (batch=32): lr=0.0001
--lr 0.0001 --batch_size 32

# Multi-GPU (batch=64): lr=0.0002 (hoặc giữ nguyên 0.0001)
--lr 0.0002 --batch_size 64 --gpu_ids "0,1"
```

### 4. **Kiểm tra GPU Usage**
Trong khi training, mở terminal khác và chạy:
```powershell
nvidia-smi -l 1
```
Bạn sẽ thấy cả GPU 0 và GPU 1 đang được sử dụng.

---

## 🔧 Cách Hoạt Động của DataParallel

1. **Model replication**: Model được copy sang cả 2 GPUs
2. **Data splitting**: Mỗi batch được chia đều cho 2 GPUs
   - Batch size = 64 → GPU 0 xử lý 32, GPU 1 xử lý 32
3. **Forward pass**: Mỗi GPU tính toán forward pass độc lập
4. **Gather outputs**: Kết quả từ 2 GPUs được tổng hợp trên GPU 0
5. **Backward pass**: Gradients được tính và đồng bộ
6. **Update**: Weights được update đồng bộ trên cả 2 GPUs

---

## 🚀 Lệnh Training Khuyến Nghị với 2 GPUs

### Phương án 1: Full Training (100 epochs)
```powershell
python train.py `
  --data_dir "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\Eggplant Dataset\Classified Images" `
  --gpu_ids "0,1" `
  --epochs 100 `
  --batch_size 64 `
  --lr 0.0001 `
  --weight_decay 0.01 `
  --loss focal_loss `
  --focal_gamma 2.0 `
  --optimizer adamw `
  --use_scheduler `
  --early_stopping 15 `
  --use_weighted_sampler `
  --num_workers 8 `
  --seed 42 `
  --output_dir "./outputs_dual_gpu_full"
```

### Phương án 2: Quick Test (5 epochs để kiểm tra)
```powershell
python train.py `
  --gpu_ids "0,1" `
  --epochs 5 `
  --batch_size 64 `
  --lr 0.0001 `
  --loss focal_loss `
  --num_workers 8 `
  --output_dir "./test_dual_gpu"
```

---

## ⚠️ Lưu ý

### Khi nào nên dùng Multi-GPU?
✅ **NÊN dùng nếu**:
- Có 2+ GPUs
- Dataset lớn (>10,000 ảnh)
- Muốn training nhanh hơn
- Muốn tăng batch size để stable hơn

❌ **KHÔNG cần nếu**:
- Chỉ có 1 GPU
- Dataset nhỏ (<5,000 ảnh)
- Đang test/debug code

### Troubleshooting

**Lỗi: "CUDA out of memory"**
→ Giảm batch size:
```powershell
--batch_size 48  # Thay vì 64
```

**Lỗi: "RuntimeError: Expected all tensors to be on the same device"**
→ Đảm bảo tất cả data đều được move to device đúng (code đã xử lý)

**GPU 1 không được sử dụng**
→ Kiểm tra lại `--gpu_ids`:
```powershell
--gpu_ids "0,1"  # Đảm bảo có dấu phẩy
```

---

## 📈 Kết Quả Mong Đợi

Với 2 GPUs, bạn sẽ thấy:
- Training speed tăng ~1.7-1.9x (không phải đúng 2x do overhead)
- Có thể train với batch size lớn hơn
- Memory usage phân bổ đều trên 2 GPUs
- Final metrics tương đương single GPU (vì model giống nhau)

---

## 🎓 So sánh DataParallel vs DistributedDataParallel

| Tiêu chí | DataParallel (DP) | DistributedDataParallel (DDP) |
|----------|-------------------|-------------------------------|
| **Dễ sử dụng** | ✅ Rất đơn giản | ⚠️ Phức tạp hơn |
| **Tốc độ** | ⭐⭐⭐ Tốt | ⭐⭐⭐⭐⭐ Rất tốt |
| **Windows support** | ✅ Tốt | ⚠️ Có thể gặp vấn đề |
| **Multi-node** | ❌ Không | ✅ Có |
| **Khuyến nghị** | ✅ Cho 1 máy, 2-4 GPUs | ✅ Cho cluster, >4 GPUs |

**Kết luận**: Với 2 GPUs trên Windows, **DataParallel** là lựa chọn tốt nhất!

---

## 📝 Summary Commands

```powershell
# Single GPU (GPU 0)
python train.py --gpu_ids "0" --batch_size 32 --num_workers 4

# Multi-GPU (GPU 0,1) - RECOMMENDED
python train.py --gpu_ids "0,1" --batch_size 64 --num_workers 8

# Chỉ GPU 1
python train.py --gpu_ids "1" --batch_size 32 --num_workers 4
```

**Lưu ý**: Nhớ thêm các tham số training khác như `--epochs`, `--lr`, `--loss`, v.v.
