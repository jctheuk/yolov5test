#!/bin/bash
# YOLOv5SC Classification Backbone Training with regurgitationV1 Dataset

echo "🚀 Starting YOLOv5SC Classification Backbone Training..."
echo ""
echo "Dataset: regurgitationV1"
echo "Architecture: Classification head connected to backbone"
echo "Training: Joint detection + classification"
echo "Epochs: 300, Batch: 128, Device: auto"
echo ""

# Start training
python train.py \
    --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5sc_classify_backbone.yaml \
    --hyp data/hyps/hyp.classify_backbone_final.yaml \
    --weights yolov5s.pt \
    --epochs 300 \
    --batch-size 128 \
    --img 416 \
    --device auto \
    --workers 8 \
    --cache ram \
    --optimizer AdamW \
    --patience 0 \
    --project runs/train-classify-backbone \
    --name exp \
    --exist-ok

echo ""
echo "✅ Training completed!"
echo "Results saved to: runs/train-classify-backbone/exp/"
echo ""
