@echo off
echo Starting Quick Test Training...

python yolov5c/train.py --data Regurgitation-YOLODataset-Detection/data.yaml --hyp yolov5c/data/hyps/hyp.fixed.yaml --epochs 5 --batch-size 16 --device cpu

echo Training completed.
pause
