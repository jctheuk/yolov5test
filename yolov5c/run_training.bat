@echo off
python train.py --data ../Regurgitation-YOLODataset-Detection/data.yaml --cfg models/yolov5sc.yaml --hyp data/hyps/hyp.custom.yaml --epochs 10 --batch-size 8 --img 416 --workers 0
pause 