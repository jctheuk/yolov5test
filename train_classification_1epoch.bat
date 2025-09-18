@echo off
REM YOLOv5 Original Classification Training - 1 Epoch Test
REM Using the dedicated classify module

echo Starting YOLOv5 Classification Training (1 Epoch Test)...
echo =====================================================

REM Switch to yolov5original directory
cd yolov5original

REM Create a simple test dataset structure first
echo Creating test dataset structure...
mkdir datasets\regurgitation-test\train\A4C 2>nul
mkdir datasets\regurgitation-test\train\PSAX 2>nul
mkdir datasets\regurgitation-test\train\PLAX 2>nul
mkdir datasets\regurgitation-test\val\A4C 2>nul
mkdir datasets\regurgitation-test\val\PSAX 2>nul
mkdir datasets\regurgitation-test\val\PLAX 2>nul

REM Copy a few test images (you can manually add some images to these folders)
echo Please add some test images to the classification folders:
echo - datasets\regurgitation-test\train\A4C\
echo - datasets\regurgitation-test\train\PSAX\
echo - datasets\regurgitation-test\train\PLAX\
echo - datasets\regurgitation-test\val\A4C\
echo - datasets\regurgitation-test\val\PSAX\
echo - datasets\regurgitation-test\val\PLAX\

REM Run classification training with 1 epoch
python classify/train.py ^
    --data datasets/regurgitation-test ^
    --model yolov5s-cls.pt ^
    --epochs 1 ^
    --batch-size 4 ^
    --imgsz 224 ^
    --device cpu ^
    --name test_1epoch ^
    --project runs/train-cls ^
    --exist-ok

echo Training completed!
pause
