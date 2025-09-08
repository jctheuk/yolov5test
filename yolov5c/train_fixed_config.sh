#!/bin/bash
# YOLOv5WithClassification 修正配置訓練腳本
# 使用診斷修正的配置，但不使用特定權重文件

echo "🚀 Starting YOLOv5WithClassification training with fixed configuration..."
echo "📊 Using diagnostic-fixed hyperparameters"
echo "🔧 Improved bias initialization applied automatically"
echo ""

cd /work/jonchang3909/yolov5test/yolov5c/ && \
python train.py \
    --data ../Regurgitation-YOLODataset-Detection/data.yaml \
    --cfg models/yolov5sc.yaml \
    --hyp data/hyps/hyp.diagnostic_fixed.yaml \
    --epochs 200 \
    --batch-size 32 \
    --img 416 \
    --save-period 10 \
    --name fixed_config_training \
    --cache \
    --patience 0 \
    --device auto

echo ""
echo "✅ Training completed with fixed configuration!"
echo "📈 Check runs/train/fixed_config_training/ for results"
