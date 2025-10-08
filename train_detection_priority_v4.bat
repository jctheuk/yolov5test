@echo off
REM Detection Priority V4 Training Script
REM Further reduced classification loss for maximum detection performance

echo ========================================
echo Detection Priority V4 Training
REM Target: Maximum detection performance with minimal classification interference
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

REM Display V4 configuration highlights
echo [2/4] V4 Configuration Highlights:
echo   Detection Loss Weights (UNCHANGED from V1):
echo     - box: 0.05 (unchanged)
echo     - cls: 0.5 (unchanged)
echo     - obj: 1.0 (unchanged)
echo   Classification Impact (FURTHER REDUCED):
echo     - cls_task: 0.01 (further reduced from V1's 0.05)
echo     - label_smoothing: 0.05 (reduced from 0.1)
echo   Other Parameters (UNCHANGED from V1):
echo     - lr0: 0.01 (unchanged)
echo     - warmup_epochs: 3.0 (unchanged)
echo     - constraint_weight: 0.5 (unchanged)
echo.

REM Start training
echo [3/4] Starting V4 training...
echo   Strategy: Further minimize classification interference
echo   Expected improvements over V1:
echo     - Even better detection performance
echo     - Minimal classification task impact
echo     - Maximum focus on detection learning
echo.

python train.py ^
    --data Regurgitation-YOLODataset-Detection/data.yaml ^
    --hyp yolov5c/data/hyps/hyp.detection_priority_v4.yaml ^
    --epochs 50 ^
    --batch-size 16 ^
    --device auto ^
    --name detection_priority_v4 ^
    --project runs/detection_priority_test

REM Check if training completed successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [4/4] V4 Training completed successfully!
    echo.
    echo ========================================
    echo V4 Results Analysis:
    echo ========================================
    echo   Results location: runs/detection_priority_test/detection_priority_v4/
    echo.
    echo   Compare with previous results:
    echo     V1 (cls_task: 0.05): mAP50 0.551, Recall 0.583, Classification 95.58%%
    echo     V4 (cls_task: 0.01): Target even better detection, Classification may drop
    echo.
    echo   Key files to review:
    echo     - results.csv (training metrics)
    echo     - classification_metrics.txt (classification performance)
    echo     - results.png (training curves)
    echo     - classification_metrics.png (classification curves)
    echo.
    echo   Expected V4 improvements:
    echo     - Detection mAP50: 0.551 → 0.60+ (target: 60%%+)
    echo     - Detection Recall: 0.583 → 0.65+ (target: 65%%+)
    echo     - Classification: 95.58%% → 90%%+ (acceptable drop)
    echo.
    echo ========================================
) else (
    echo.
    echo [ERROR] V4 Training failed with error code %ERRORLEVEL%
    echo Please check the error messages above and try again.
    echo.
)

echo Press any key to exit...
pause >nul


