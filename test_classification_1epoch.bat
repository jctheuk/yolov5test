@echo off
REM YOLOv5 Classification Test - 1 Epoch
REM Test with imgsz 416, batch 4

echo Starting YOLOv5 Classification Test (1 Epoch)...
echo ===============================================

REM Switch to yolov5original directory
cd yolov5original

REM Run classification training with 1 epoch and specified settings
python classify/train.py ^
    --data datasets/regurgitationV1-cls ^
    --model yolov5s-cls.pt ^
    --epochs 1 ^
    --batch-size 4 ^
    --imgsz 416 ^
    --device cpu ^
    --workers 0 ^
    --name test_1epoch ^
    --project runs/train-cls ^
    --exist-ok

echo Test completed!
pause