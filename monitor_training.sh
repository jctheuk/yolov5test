#!/bin/bash
# 監控腳本 - 實時監控訓練狀態

echo "開始監控訓練..."

# 監控最新的日誌文件
tail -f yolov5c/runs/train/exp/train.log | grep -E "(Epoch|DEBUG|WARNING|ERROR|mAP|Accuracy)"
