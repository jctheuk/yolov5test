#!/bin/bash
# 驗證腳本 - 檢查檢測結果輸出

echo "開始驗證..."

# 使用最新的權重文件進行驗證
python yolov5c/val.py \
    --weights yolov5c/runs/train/exp/weights/best.pt \
    --data Regurgitation-YOLODataset-Detection/data.yaml \
    --verbose \
    --save-txt \
    --save-conf

echo "驗證完成！"
