@echo off
echo Testing YOLOv5 Classification Training with Validation Fix
echo ========================================================

REM Clear dataset caches before training
echo Clearing dataset caches...
set DATASET=../Regurgitation-YOLODataset-Detection
for %%d in (train valid test) do (
    if exist "%DATASET%\%%d\labels\labels.cache" del "%DATASET%\%%d\labels\labels.cache"
    if exist "%DATASET%\%%d\labels\labels.cache.npy" del "%DATASET%\%%d\labels\labels.cache.npy"
    if exist "%DATASET%\%%d\labels\labels_cl.cache.npy" del "%DATASET%\%%d\labels\labels_cl.cache.npy"
    for %%f in ("%DATASET%\%%d\labels\*.cache*") do del "%%f" 2>nul
)

echo Starting training with validation fix...
python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --epochs 5 ^
    --batch-size 8 ^
    --device auto ^
    --patience 0 ^
    --project runs/test_fix ^
    --name validation_fix_test

echo Training completed!
pause
