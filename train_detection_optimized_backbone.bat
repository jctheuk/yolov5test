@echo off
REM Detection-Optimized Training Script for Backbone-Connected Classification Model
REM Target: mAP50 70-80% while maintaining classification performance

echo ========================================
echo Detection mAP 70-80%% Optimization Training
echo ========================================
echo.

REM Clear dataset caches for consistent results
echo [1/4] Clearing dataset caches...
set DATASET=Regurgitation-YOLODataset-Detection
set sets=train valid test
for %%d in (%sets%) do (
    set labels=%DATASET%\%%d\labels
    if exist "%labels%\*.cache*" (
        del /q "%labels%\*.cache*" 2>nul
        echo   Cleared cache files in %%d\labels
    )
)
echo   Dataset caches cleared successfully
echo.

REM Display configuration
echo [2/4] Training Configuration:
echo   Model: yolov5sc_classify_backbone.yaml
echo   Hyperparameters: hyp.detection_optimized_backbone.yaml
echo   Target: mAP50 70-80%%
echo   Epochs: 50
echo   Batch Size: 16
echo   Device: auto
echo.

REM Start training
echo [3/4] Starting optimized training...
echo   This may take 2-4 hours depending on your hardware
echo.

python train.py ^
    --data Regurgitation-YOLODataset-Detection/data.yaml ^
    --hyp yolov5c/data/hyps/hyp.detection_optimized_backbone.yaml ^
    --epochs 50 ^
    --batch-size 16 ^
    --device auto ^
    --name detection_optimized_backbone ^
    --project runs/detection_optimization

REM Check if training completed successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [4/4] Training completed successfully!
    echo.
    echo ========================================
    echo Results Analysis:
    echo ========================================
    echo   Check results in: runs/detection_optimization/detection_optimized_backbone/
    echo   Key files to review:
    echo     - results.csv (training metrics)
    echo     - classification_metrics.txt (classification performance)
    echo     - results.png (training curves)
    echo     - classification_metrics.png (classification curves)
    echo.
    echo Expected improvements:
    echo   - mAP50: 0.551 → 0.70+ (target: 70-80%%)
    echo   - mAP50-95: 0.209 → 0.30+ (target: 30-40%%)
    echo   - Recall: 0.583 → 0.70+ (target: 70-80%%)
    echo   - Classification: 95.58%% → 90%%+ (maintain medical standards)
    echo.
    echo ========================================
) else (
    echo.
    echo [ERROR] Training failed with error code %ERRORLEVEL%
    echo Please check the error messages above and try again.
    echo.
)

echo Press any key to exit...
pause >nul

