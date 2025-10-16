#!/bin/bash
# YOLOv5 Original Classification Training - Medium-Large Model
# Using custom yolov5ml.yaml configuration
# Batch size: 48 (optimized for ML model)
# Training from scratch with custom architecture

# ========== YOLOv5ML Classification - V1 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV1-Classification --model models/yolov5ml.yaml --epochs 300 --batch-size 48 --imgsz 416 --name classifyml_v1 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4 --cutoff 10 --pretrained False

# ========== YOLOv5ML Classification - V2 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV2-Classification --model models/yolov5ml.yaml --epochs 300 --batch-size 48 --imgsz 416 --name classifyml_v2 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4 --cutoff 10 --pretrained False

# ========== YOLOv5ML Classification - V3 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV3-Classification --model models/yolov5ml.yaml --epochs 300 --batch-size 48 --imgsz 416 --name classifyml_v3 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4 --cutoff 10 --pretrained False

# ========== YOLOv5ML Classification - V4 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV4-Classification --model models/yolov5ml.yaml --epochs 300 --batch-size 48 --imgsz 416 --name classifyml_v4 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4 --cutoff 10 --pretrained False

# ========== YOLOv5ML Classification - V5 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV5-Classification --model models/yolov5ml.yaml --epochs 300 --batch-size 48 --imgsz 416 --name classifyml_v5 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4 --cutoff 10 --pretrained False

