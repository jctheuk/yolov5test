@echo off
REM YOLOv5 Original Classification Training
REM Using the dedicated classify module

echo Starting YOLOv5 Classification Training...
echo ==========================================

REM Switch to yolov5original directory
cd yolov5original

REM Run classification training
python classify/train.py ^
    --data datasets/regurgitation-classification ^
    --model yolov5s-cls.pt ^
    --epochs 10 ^
    --batch-size 16 ^
    --imgsz 224 ^
    --device cpu ^
    --name regurgitation_classification ^
    --project runs/train-cls ^
    --exist-ok

echo Training completed!
pause
