@echo off
REM YOLOv5WithClassification 記憶體優化版本
REM 針對記憶體不足問題的解決方案

echo Starting YOLOv5WithClassification Memory Optimized Training...
echo ============================================================

REM 清理資料集快取
echo Cleaning dataset caches...
powershell -Command "$DATASET = 'Regurgitation-YOLODataset-Detection'; $sets = @('train', 'valid', 'test'); foreach ($d in $sets) { $labels = Join-Path (Join-Path $DATASET $d) 'labels'; Remove-Item -Path (Join-Path $labels 'labels.cache') -ErrorAction SilentlyContinue -Force; Remove-Item -Path (Join-Path $labels 'labels.cache.npy') -ErrorAction SilentlyContinue -Force; Remove-Item -Path (Join-Path $labels 'labels_cl.cache.npy') -ErrorAction SilentlyContinue -Force; Get-ChildItem -Path $labels -Filter '*.cache*' -ErrorAction SilentlyContinue | Remove-Item -Force }"

REM 切換到 yolov5c 目錄
cd yolov5c

REM 運行記憶體優化訓練
python train.py ^
    --data ../Regurgitation-YOLODataset-Detection/data.yaml ^
    --cfg models/yolov5sc.yaml ^
    --epochs 1 ^
    --batch-size 1 ^
    --imgsz 416 ^
    --device cpu ^
    --workers 0 ^
    --patience 0 ^
    --hyp data/hyps/hyp.fixed.yaml ^
    --name memory_optimized_test ^
    --project runs/train ^
    --exist-ok

echo Training completed!
pause
