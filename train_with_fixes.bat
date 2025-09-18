@echo off
echo YOLOv5WithClassification 修復版訓練腳本
echo ================================================

echo 清理數據集快取...
del /q Regurgitation-YOLODataset-Detection\train\labels\*.cache* 2>nul
del /q Regurgitation-YOLODataset-Detection\valid\labels\*.cache* 2>nul
del /q Regurgitation-YOLODataset-Detection\test\labels\*.cache* 2>nul

echo 開始修復版訓練...
python yolov5c/train.py ^
    --data Regurgitation-YOLODataset-Detection/data.yaml ^
    --hyp yolov5c/data/hyps/hyp.fixed.yaml ^
    --epochs 50 ^
    --batch-size 16 ^
    --device auto ^
    --patience 10 ^
    --min-delta 0.001 ^
    --verbose

echo 訓練完成！
echo 檢查結果...
python check_train_log_output.py

pause
