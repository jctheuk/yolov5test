#!/bin/bash
# YOLOv5WithClassification 修復版訓練腳本

echo "開始修復版訓練..."

# 清理快取
echo "清理數據集快取..."
rm -f Regurgitation-YOLODataset-Detection/train/labels/*.cache*
rm -f Regurgitation-YOLODataset-Detection/valid/labels/*.cache*
rm -f Regurgitation-YOLODataset-Detection/test/labels/*.cache*

# 使用修復的超參數進行訓練
echo "開始訓練..."
python yolov5c/train.py \
    --data Regurgitation-YOLODataset-Detection/data.yaml \
    --hyp yolov5c/data/hyps/hyp.fixed.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --patience 10 \
    --min-delta 0.001 \
    --verbose

echo "訓練完成！"
