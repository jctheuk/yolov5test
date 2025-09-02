#!/bin/bash

echo "Starting YOLOv5 training with ORIGINAL learning rate (fix NaN issues)..."
echo ""
echo "Training configuration:"
echo "- Dataset: Regurgitation-YOLODataset-Detection"
echo "- Model: YOLOv5s with Classification"
echo "- Epochs: 300"
echo "- Early stopping: DISABLED (patience=0)"
echo "- Batch size: 128 (as requested)"
echo "- Image size: 416"
echo "- Classification: ENABLED (very low weights)"
echo "- Learning rate: KEPT ORIGINAL (0.01 as requested)"
echo "- Gradient clipping: ENHANCED (max_norm=1.0)"
echo "- Data augmentation: DISABLED (medical images)"
echo "- Cache: ENABLED"
echo ""

# Clear dataset caches before training
echo "Clearing dataset caches..."
rm -f ../Regurgitation-YOLODataset-Detection/train/labels/*.cache*
rm -f ../Regurgitation-YOLODataset-Detection/valid/labels/*.cache*
rm -f ../Regurgitation-YOLODataset-Detection/test/labels/*.cache*
echo "Cache clearing completed."
echo ""

python train.py \
    --data ../Regurgitation-YOLODataset-Detection/data.yaml \
    --cfg models/yolov5sc.yaml \
    --hyp data/hyps/hyp.fix_nan_keep_lr.yaml \
    --epochs 300 \
    --batch-size 128 \
    --img 416 \
    --workers 0 \
    --patience 0 \
    --save-period 10 \
    --name 300_epochs_keep_lr_fix_nan \
    --cache

echo ""
echo "Training completed!"
