#!/bin/bash

# TWCC Training Script for YOLOv5WithClassification
# 關閉早停，獲得完整訓練圖表

# 載入必要的模組
module load python/3.8
module load cuda/11.0

# 啟動 conda 環境
source ~/.bashrc
conda activate base

# 檢查 Python 路徑
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# 清理資料快取
echo "Clearing dataset caches..."
DATASET="Regurgitation-YOLODataset-Detection"
sets=("train" "valid" "test")
for d in "${sets[@]}"; do
    labels="${DATASET}/${d}/labels"
    if [ -d "$labels" ]; then
        rm -f "${labels}/labels.cache"
        rm -f "${labels}/labels.cache.npy"
        rm -f "${labels}/labels_cl.cache.npy"
        rm -f "${labels}"/*.cache*
        echo "Cleared caches in ${labels}"
    fi
done

# 開始訓練 - 關閉早停，獲得完整訓練圖表
echo "Starting training with classification enabled..."
python train.py \
    --data Regurgitation-YOLODataset-Detection/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --patience 0

echo "Training completed!"
