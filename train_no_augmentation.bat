@echo off
echo Starting YOLOv5 training with NO AUGMENTATION for medical images...
echo.

REM Clear dataset caches before training
echo Clearing dataset caches...
set DATASET=Regurgitation-YOLODataset-Detection
for %%d in (train valid test) do (
    if exist "%DATASET%\%%d\labels\labels.cache" del "%DATASET%\%%d\labels\labels.cache"
    if exist "%DATASET%\%%d\labels\labels.cache.npy" del "%DATASET%\%%d\labels\labels.cache.npy"
    if exist "%DATASET%\%%d\labels\labels_cl.cache.npy" del "%DATASET%\%%d\labels\labels_cl.cache.npy"
    for %%f in ("%DATASET%\%%d\labels\*.cache*") do del "%%f"
)
echo Dataset caches cleared.

echo.
echo Starting training with NO AUGMENTATION...
echo Using hyperparameters: hyp.no_augmentation_medical.yaml
echo.

cd yolov5c
python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --hyp data/hyps/hyp.no_augmentation_medical.yaml ^
    --epochs 50 ^
    --batch-size 16 ^
    --device auto ^
    --name no_augmentation_medical ^
    --patience 0

echo.
echo Training completed!
pause
