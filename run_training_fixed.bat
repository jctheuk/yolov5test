@echo off
echo Starting YOLOv5 Classification Training (Fixed)...
echo ================================================

cd yolov5original
python classify/train.py --data datasets/regurgitationV1-cls --model yolov5s-cls.pt --epochs 1 --batch-size 2 --imgsz 416 --device cpu --workers 0 --name test_fixed --project runs/train-cls --exist-ok --nosave

echo Training completed!
pause

