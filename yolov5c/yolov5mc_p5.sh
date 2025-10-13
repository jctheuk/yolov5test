#!/bin/bash
# YOLOv5mc P5 Configuration - K-Fold Training V1-V5
# TWCC.ai Training Script

echo "=== Starting YOLOv5mc P5 K-Fold Training V1-V5 ==="
echo "Start time: $(date)"

cd /work/jonchang3909/yolov5test/yolov5c/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && echo "=== FOLD 1 - V1 ===" && echo "V1 start: $(date)" && python train.py --data ../regurgitationV1/data.yaml --cfg models/yolov5mc_p5.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mc_p5_v1 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml && echo "V1 end: $(date)"

echo "=== FOLD 2 - V2 ===" && echo "V2 start: $(date)" && python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5mc_p5.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mc_p5_v2 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml && echo "V2 end: $(date)"

echo "=== FOLD 3 - V3 ===" && echo "V3 start: $(date)" && python train.py --data ../regurgitationV3/data.yaml --cfg models/yolov5mc_p5.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mc_p5_v3 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml && echo "V3 end: $(date)"

echo "=== FOLD 4 - V4 ===" && echo "V4 start: $(date)" && python train.py --data ../regurgitationV4/data.yaml --cfg models/yolov5mc_p5.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mc_p5_v4 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml && echo "V4 end: $(date)"

echo "=== FOLD 5 - V5 ===" && echo "V5 start: $(date)" && python train.py --data ../regurgitationV5/data.yaml --cfg models/yolov5mc_p5.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mc_p5_v5 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml && echo "V5 end: $(date)"

echo "=== YOLOv5mc P5 K-Fold Training Complete ==="
echo "End time: $(date)"
echo "Check runs/train/ for results"
