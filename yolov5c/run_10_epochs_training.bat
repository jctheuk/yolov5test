@echo off
echo Starting YOLOv5 training for 10 epochs...
echo.
echo Training configuration:
echo - Dataset: Regurgitation-YOLODataset-Detection
echo - Model: YOLOv5s with Classification
echo - Epochs: 10
echo - Early stopping: DISABLED (patience=0)
echo - Batch size: 8
echo - Image size: 416
echo - Classification: ENABLED (one-hot encoding)
echo.

python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --hyp data/hyps/hyp.fixed_classification.yaml ^
    --epochs 10 ^
    --batch-size 8 ^
    --img 416 ^
    --workers 0 ^
    --patience 0 ^
    --save-period 2 ^
    --name 10_epochs_training

echo.
echo Training completed!
pause
