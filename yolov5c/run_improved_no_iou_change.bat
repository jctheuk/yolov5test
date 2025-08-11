@echo off
echo Starting improved YOLOv5 training with optimized parameters (IoU unchanged)...
echo.
echo Key improvements (IoU threshold kept at 0.20):
echo - Learning rate increased to 0.01 (from 0.001)
echo - Training for 50 epochs (from 10)
echo - Early stopping disabled for complete training curves
echo - Balanced data augmentation for medical images
echo - Optimized loss weights for better multi-task learning
echo - Reduced warmup epochs for faster convergence
echo.

python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --hyp data/hyps/hyp.improved_no_iou_change.yaml ^
    --epochs 50 ^
    --batch-size 8 ^
    --img 416 ^
    --workers 0 ^
    --patience 0 ^
    --save-period 10

echo.
echo Training completed!
pause
