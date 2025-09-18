#!/bin/bash
# YOLOv5SC Classification Backbone Cloud Training Script

echo "🚀 Starting YOLOv5SC Classification Backbone Training on Cloud..."
echo ""
echo "Configuration: yolov5sc_classify_backbone.yaml"
echo "Dataset: regurgitationV1"
echo "Architecture: Classification head connected to backbone"
echo "Training: Joint detection + classification"
echo "Epochs: 300, Batch: 128, Device: auto"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Check dataset
echo "📁 Checking dataset..."
ls -la ../regurgitationV1/

# Start training
echo "🏃 Starting training..."
python train.py \
    --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5sc_classify_backbone.yaml \
    --hyp data/hyps/hyp.classify_backbone.yaml \
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
echo "📊 Key files:"
echo "  - Best model: runs/train-classify-backbone/exp/weights/best.pt"
echo "  - Last model: runs/train-classify-backbone/exp/weights/last.pt"
echo "  - Results: runs/train-classify-backbone/exp/results.txt"
echo "  - Logs: runs/train-classify-backbone/exp/train.log"
echo ""
