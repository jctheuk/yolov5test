@echo off
echo Starting YOLOv5 training for 10 epochs (Optimized Pooling)...
echo.
echo Training configuration:
echo - Dataset: Regurgitation-YOLODataset-Detection
echo - Model: YOLOv5s with Classification
echo - Epochs: 10
echo - Early stopping: DISABLED (patience=0)
echo - Batch size: 8
echo - Image size: 416
echo - Classification: ENABLED (one-hot encoding)
echo - Pooling: OPTIMIZED (single AdaptiveAvgPool2d)
echo - CUDA Error Fix: ENABLED (no AdaptiveMaxPool2d)
echo.

REM Clear dataset caches before training
echo Clearing dataset caches...
if exist "..\Regurgitation-YOLODataset-Detection\train\labels\*.cache*" (
    del /Q "..\Regurgitation-YOLODataset-Detection\train\labels\*.cache*" 2>nul
)
if exist "..\Regurgitation-YOLODataset-Detection\valid\labels\*.cache*" (
    del /Q "..\Regurgitation-YOLODataset-Detection\valid\labels\*.cache*" 2>nul
)
if exist "..\Regurgitation-YOLODataset-Detection\test\labels\*.cache*" (
    del /Q "..\Regurgitation-YOLODataset-Detection\test\labels\*.cache*" 2>nul
)
echo Cache clearing completed.
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
    --name 10_epochs_training_optimized

echo.
echo Training completed!
pause
