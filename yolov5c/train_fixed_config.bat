@echo off
echo 🚀 Starting YOLOv5WithClassification training with fixed configuration...
echo 📊 Using diagnostic-fixed hyperparameters
echo 🔧 Improved bias initialization applied automatically
echo.

cd /d "%~dp0"

python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --hyp data/hyps/hyp.diagnostic_fixed.yaml ^
    --epochs 200 ^
    --batch-size 32 ^
    --img 416 ^
    --save-period 10 ^
    --name fixed_config_training ^
    --cache ^
    --patience 0 ^
    --device auto

echo.
echo ✅ Training completed with fixed configuration!
echo 📈 Check runs/train/fixed_config_training/ for results
pause
