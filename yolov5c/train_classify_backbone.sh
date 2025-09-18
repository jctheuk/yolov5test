#!/bin/bash
# YOLOv5SC with Classification Head Connected to Backbone Training Script

echo "🚀 Starting YOLOv5SC with Classification Head Connected to Backbone Training..."
echo ""
echo "Configuration: yolov5sc_classify_backbone.yaml"
echo "Dataset: Regurgitation-YOLODataset-Detection"
echo "Classification: Connected to backbone (like classify/)"
echo ""

python train.py \
    --data ../Regurgitation-YOLODataset-Detection/data.yaml \
    --cfg models/yolov5sc_classify_backbone.yaml \
    --weights yolov5s.pt \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --project runs/train-classify-backbone \
    --name exp \
    --exist-ok

echo ""
echo "✅ Training completed!"
echo "Results saved to: runs/train-classify-backbone/exp/"
echo ""
