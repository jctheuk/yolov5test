#!/bin/bash

echo "Starting YOLOv5 STABLE training for 300 epochs (Fix NaN issues)..."
echo ""
echo "Training configuration:"
echo "- Dataset: Regurgitation-YOLODataset-Detection"
echo "- Model: YOLOv5s with Classification"
echo "- Epochs: 300"
echo "- Early stopping: DISABLED (patience=0)"
echo "- Batch size: 16 (reduced from 32 for stability)"
echo "- Image size: 416"
echo "- Classification: ENABLED (very low weights)"
echo "- Learning rate: REDUCED (0.0005 to prevent explosion)"
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
    --hyp data/hyps/hyp.stable_300_epochs.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img 416 \
    --workers 0 \
    --patience 0 \
    --save-period 10 \
    --name 300_epochs_stable_fix_nan \
    --cache

echo ""
echo "Training completed!"
