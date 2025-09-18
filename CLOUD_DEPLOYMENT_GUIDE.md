# YOLOv5 Classification Cloud Deployment Guide

## 🚀 **How to Duplicate Success on Cloud**

Based on the successful local training, here's how to deploy the same configuration on cloud platforms.

## 📋 **Successful Configuration**

```bash
python classify/train.py \
    --data datasets/regurgitationV1-cls \
    --model yolov5s-cls.pt \
    --epochs 1 \
    --batch-size 2 \
    --imgsz 416 \
    --device cpu \
    --workers 0 \
    --name test_batch2 \
    --project runs/train-cls \
    --exist-ok
```

## ☁️ **Cloud Platform Options**

### **1. Google Colab (Free GPU)**

```python
# Setup
!git clone https://github.com/ultralytics/yolov5.git
!cd yolov5 && pip install -r requirements.txt

# Upload your dataset
from google.colab import files
files.upload()  # Upload regurgitationV1-cls.zip

# Extract dataset
!unzip regurgitationV1-cls.zip -d yolov5/datasets/

# Run training
!cd yolov5 && python classify/train.py \
    --data datasets/regurgitationV1-cls \
    --model yolov5s-cls.pt \
    --epochs 10 \
    --batch-size 16 \
    --imgsz 416 \
    --device 0 \
    --workers 4 \
    --name cloud_training \
    --project runs/train-cls \
    --exist-ok
```

### **2. AWS EC2**

```bash
# Launch EC2 instance (g4dn.xlarge for GPU)
# Install dependencies
sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision torchaudio
pip3 install ultralytics

# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Upload dataset (use S3 or scp)
aws s3 cp s3://your-bucket/regurgitationV1-cls.zip ./
unzip regurgitationV1-cls.zip -d datasets/

# Run training
python classify/train.py \
    --data datasets/regurgitationV1-cls \
    --model yolov5s-cls.pt \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 416 \
    --device 0 \
    --workers 8 \
    --name aws_training \
    --project runs/train-cls \
    --exist-ok
```

### **3. Azure ML**

```python
# Azure ML Notebook
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

# Create environment
env = Environment.from_conda_specification(
    name="yolov5-env",
    file_path="environment.yml"
)

# Create script config
script_config = ScriptRunConfig(
    source_directory="./yolov5",
    script="classify/train.py",
    arguments=[
        "--data", "datasets/regurgitationV1-cls",
        "--model", "yolov5s-cls.pt",
        "--epochs", "50",
        "--batch-size", "32",
        "--imgsz", "416",
        "--device", "0",
        "--workers", "8",
        "--name", "azure_training",
        "--project", "runs/train-cls",
        "--exist-ok"
    ],
    environment=env,
    compute_target="gpu-cluster"
)

# Submit experiment
experiment = Experiment(workspace=ws, name="yolov5-classification")
run = experiment.submit(script_config)
```

### **4. Google Cloud AI Platform**

```bash
# Setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create training job
gcloud ai-platform jobs submit training yolov5_classification \
    --package-path ./yolov5 \
    --module-name classify.train \
    --region us-central1 \
    --python-version 3.8 \
    --runtime-version 2.8 \
    --scale-tier BASIC_GPU \
    -- \
    --data datasets/regurgitationV1-cls \
    --model yolov5s-cls.pt \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 416 \
    --device 0 \
    --workers 8 \
    --name gcp_training \
    --project runs/train-cls \
    --exist-ok
```

## 🔧 **Cloud-Specific Optimizations**

### **GPU Acceleration**
```bash
# Use GPU instead of CPU
--device 0  # or 0,1,2,3 for multi-GPU

# Increase batch size for GPU
--batch-size 32  # or 64, 128 depending on GPU memory

# Use more workers
--workers 8  # or 16 for faster data loading
```

### **Memory Optimization**
```bash
# For limited GPU memory
--batch-size 16
--imgsz 224  # smaller image size

# For large datasets
--cache ram  # cache images in RAM
--workers 4  # fewer workers to save memory
```

### **Performance Tuning**
```bash
# Mixed precision training
--amp  # automatic mixed precision

# Optimized learning rate
--lr0 0.01  # higher learning rate for GPU

# Data augmentation
--augment  # enable augmentation
```

## 📦 **Dataset Preparation for Cloud**

### **1. Create Dataset Archive**
```bash
# Compress dataset
zip -r regurgitationV1-cls.zip yolov5original/datasets/regurgitationV1-cls/

# Upload to cloud storage
aws s3 cp regurgitationV1-cls.zip s3://your-bucket/
```

### **2. Dataset Structure**
```
regurgitationV1-cls/
├── train/
│   ├── A4C/
│   ├── PSAX/
│   └── PLAX/
├── val/
│   ├── A4C/
│   ├── PSAX/
│   └── PLAX/
└── test/
    ├── A4C/
    ├── PSAX/
    └── PLAX/
```

## 🚀 **Quick Start Commands**

### **Google Colab (Recommended for Testing)**
```python
# 1. Clone YOLOv5
!git clone https://github.com/ultralytics/yolov5.git
!cd yolov5 && pip install -r requirements.txt

# 2. Upload dataset
from google.colab import files
files.upload()  # Upload your dataset zip

# 3. Extract and train
!unzip regurgitationV1-cls.zip -d yolov5/datasets/
!cd yolov5 && python classify/train.py --data datasets/regurgitationV1-cls --model yolov5s-cls.pt --epochs 10 --batch-size 16 --imgsz 416 --device 0 --name colab_training --project runs/train-cls --exist-ok
```

### **AWS EC2 (Production)**
```bash
# 1. Launch GPU instance
# 2. Install dependencies
pip3 install torch torchvision ultralytics

# 3. Clone and setup
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 4. Upload dataset and train
aws s3 cp s3://your-bucket/regurgitationV1-cls.zip ./
unzip regurgitationV1-cls.zip -d datasets/
python classify/train.py --data datasets/regurgitationV1-cls --model yolov5s-cls.pt --epochs 50 --batch-size 32 --imgsz 416 --device 0 --workers 8 --name aws_training --project runs/train-cls --exist-ok
```

## 📊 **Expected Cloud Performance**

| Platform | GPU | Batch Size | Epochs | Time | Cost |
|----------|-----|------------|--------|------|------|
| Google Colab | T4 | 16 | 10 | ~30 min | Free |
| AWS EC2 | V100 | 32 | 50 | ~2 hours | ~$3 |
| Azure ML | V100 | 32 | 50 | ~2 hours | ~$3 |
| GCP AI | T4 | 16 | 50 | ~4 hours | ~$2 |

## 🎯 **Success Metrics**

- **Training Accuracy**: >80%
- **Validation Accuracy**: >75%
- **Training Time**: <2 hours for 50 epochs
- **Model Size**: <50MB
- **Inference Speed**: <100ms per image

---
**Guide Created**: 2024-12-16  
**Based on**: Successful local training with batch size 2  
**Next Step**: Choose cloud platform and deploy

