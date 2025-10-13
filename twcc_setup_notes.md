# TWCC.ai Setup Notes for YOLOv5WithClassification

## 🚀 Quick Start on TWCC.ai

### 1. **Check GPU Resources**
```bash
# Check available GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

### 2. **Adjust Batch Size Based on GPU Memory**
Edit `train_kfold_timing.sh`:

| GPU Memory | Recommended batch_size |
|------------|------------------------|
| 8GB | `batch_size=8` |
| 16GB | `batch_size=16` |
| 24GB | `batch_size=24` |
| 32GB+ | `batch_size=32` |

### 3. **Install Dependencies (if needed)**
```bash
# Update system
sudo apt-get update

# Install OpenCV dependencies  
sudo apt-get install libgl1-mesa-glx libglib2.0-0 -y

# Install Python packages
pip install -r yolov5c/requirements.txt
```

### 4. **Run Training**
```bash
# Make executable
chmod +x train_kfold_timing.sh

# Run K-fold training
./train_kfold_timing.sh
```

### 5. **Monitor Training**
```bash
# Check GPU usage during training
watch -n 1 nvidia-smi

# Monitor logs in real-time
tail -f kfold_training_timing_*.log
```

## 🔧 **TWCC.ai Specific Settings**

### Recommended Configuration:
```bash
# In train_kfold_timing.sh, adjust these based on your GPU:
epochs=50          # Good for medical datasets
batch_size=16      # Adjust based on GPU memory
hyp_file="data/hyps/hyp.default.yaml"  # Our optimized hyperparameters
```

### Expected Training Times on TWCC.ai:
| GPU Type | Batch Size | Time per Fold | Total K-Fold |
|----------|------------|---------------|--------------|
| V100 (16GB) | 16 | ~45 min | ~3.8 hours |
| V100 (32GB) | 32 | ~30 min | ~2.5 hours |
| A100 (40GB) | 32 | ~25 min | ~2.1 hours |

## 📊 **Output Files**
Training will create:
- `kfold_training_timing_YYYYMMDD_HHMMSS.log` - Complete log
- `kfold_training_summary_YYYYMMDD_HHMMSS.log` - Summary table
- `yolov5c/runs/train/fold1/`, `fold2/`, etc. - Individual model results

## 🎯 **Ready for TWCC.ai!**
Your setup is optimized for Taiwan Computing Cloud with:
- ✅ Linux bash script
- ✅ GPU-optimized batch sizes  
- ✅ Clean dataset (no constraints needed)
- ✅ Medical-specific hyperparameters
- ✅ Comprehensive timing and logging
