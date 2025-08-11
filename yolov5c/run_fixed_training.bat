@echo off
echo Starting YOLOv5 training with FIXED classification labels...
echo.
echo Key fixes and improvements:
echo - Fixed missing classification labels
echo - Learning rate increased to 0.01 (from 0.001)
echo - Training for 100 epochs (extended for better convergence)
echo - Early stopping disabled for complete training curves
echo - Balanced data augmentation for medical images
echo - Optimized loss weights for multi-task learning
echo - Reduced classification weight to 0.2 (was 0.3)
echo.

python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --hyp data/hyps/hyp.fixed_classification.yaml ^
    --epochs 100 ^
    --batch-size 8 ^
    --img 416 ^
    --workers 0 ^
    --patience 0 ^
    --save-period 10 ^
    --name fixed_classification_training

echo.
echo Training completed!
pause
