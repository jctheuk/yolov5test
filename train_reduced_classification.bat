@echo off
echo ========================================
echo YOLOv5 Reduced Classification Training
echo ========================================

echo.
echo Step 1: Clearing dataset caches...
cd /d "%~dp0"

REM Clear dataset caches before training
set DATASET=Regurgitation-YOLODataset-Detection
for %%d in (train valid test) do (
    if exist "%DATASET%\%%d\labels\labels.cache" (
        del /f /q "%DATASET%\%%d\labels\labels.cache"
        echo Deleted labels.cache from %%d
    )
    if exist "%DATASET%\%%d\labels\labels.cache.npy" (
        del /f /q "%DATASET%\%%d\labels\labels.cache.npy"
        echo Deleted labels.cache.npy from %%d
    )
    if exist "%DATASET%\%%d\labels\labels_cl.cache.npy" (
        del /f /q "%DATASET%\%%d\labels\labels_cl.cache.npy"
        echo Deleted labels_cl.cache.npy from %%d
    )
    for %%f in ("%DATASET%\%%d\labels\*.cache*") do (
        del /f /q "%%f" 2>nul
        echo Deleted cache file from %%d
    )
)

echo.
echo Step 2: Starting training with reduced classification weight...
echo Using cls_task: 0.25 (reduced from 0.5)

cd yolov5c
python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --epochs 50 ^
    --batch-size 16 ^
    --device auto ^
    --hyp data/hyps/hyp.fixed.yaml ^
    --patience 0 ^
    --project ../runs ^
    --name reduced_classification_training

echo.
echo Training completed!
echo Check results in: ../runs/reduced_classification_training/
pause
