@echo off
REM YOLOv5WithClassification 聯合訓練腳本 - 測試版本
REM 使用 CPU，1 epoch 測試

echo Starting YOLOv5WithClassification Joint Training (Test)...
echo ========================================================

REM 切換到 yolov5c 目錄
cd yolov5c

REM 運行訓練 - 使用 CPU，1 epoch 測試
python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --epochs 1 ^
    --batch-size 2 ^
    --device cpu ^
    --patience 0 ^
    --hyp data/hyps/hyp.fixed.yaml ^
    --name classification_test ^
    --project runs/train ^
    --exist-ok

echo Training completed!
pause