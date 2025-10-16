#!/bin/bash
# YOLOv5MLC P4 Configuration - K-Fold Training V1-V5
# MLC = Medium-Large Classification (depth=0.83, width=0.875)
# TWCC.ai Training Script

echo "=== Starting YOLOv5MLC P4 K-Fold Training V1-V5 ==="
echo "Start time: $(date)"

cd /work/jonchang3909/yolov5test/yolov5c/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && echo "=== FOLD 1 - V1 ===" && echo "V1 start: $(date)" && python train.py --data ../regurgitationV1/data.yaml --cfg models/yolov5mlc_p4.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mlc_p4_v1 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml  && echo "V1 end: $(date)"

echo "=== FOLD 2 - V2 ===" && echo "V2 start: $(date)" && python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5mlc_p4.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mlc_p4_v2 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml  && echo "V2 end: $(date)"

echo "=== FOLD 3 - V3 ===" && echo "V3 start: $(date)" && python train.py --data ../regurgitationV3/data.yaml --cfg models/yolov5mlc_p4.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mlc_p4_v3 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml  && echo "V3 end: $(date)"

echo "=== FOLD 4 - V4 ===" && echo "V4 start: $(date)" && python train.py --data ../regurgitationV4/data.yaml --cfg models/yolov5mlc_p4.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mlc_p4_v4 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml  && echo "V4 end: $(date)"

echo "=== FOLD 5 - V5 ===" && echo "V5 start: $(date)" && python train.py --data ../regurgitationV5/data.yaml --cfg models/yolov5mlc_p4.yaml --epochs 300 --batch-size 128 --imgsz 416 --name yolov5mlc_p4_v5 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml  && echo "V5 end: $(date)"

echo "=== YOLOv5MLC P4 K-Fold Training Complete ==="
echo "End time: $(date)"
echo "Check runs/train/ for results"

