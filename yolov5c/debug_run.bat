@echo off
cd /d "%~dp0"
python train.py --data ../Regurgitation-YOLODataset-Detection/data.yaml --cfg models/yolov5sc.yaml --hyp data/hyps/hyp.fixed_classification.yaml --epochs 1 --batch-size 16 --img 416 --name debug_analysis
pause
