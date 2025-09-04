@echo off
cd yolov5c
python train.py --data ../Regurgitation-YOLODataset-Detection/data.yaml --cfg models/yolov5sc.yaml --hyp data/hyps/hyp.fixed_classification_minimal.yaml --epochs 3 --batch-size 8 --img 416 --save-period 2 --name testingv1 --cache
pause
