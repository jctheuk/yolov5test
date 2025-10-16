#!/bin/bash
# YOLOv5 Original Classification Training - Medium Model
# Using pretrained yolov5m-cls.pt
# Batch size: 64 (optimized for medium model)

# ========== YOLOv5M Classification - V1 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV1-Classification --model yolov5m-cls.pt --epochs 300 --batch-size 64 --imgsz 416 --name classifym_v1 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5M Classification - V2 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV2-Classification --model yolov5m-cls.pt --epochs 300 --batch-size 64 --imgsz 416 --name classifym_v2 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5M Classification - V3 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV3-Classification --model yolov5m-cls.pt --epochs 300 --batch-size 64 --imgsz 416 --name classifym_v3 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5M Classification - V4 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV4-Classification --model yolov5m-cls.pt --epochs 300 --batch-size 64 --imgsz 416 --name classifym_v4 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5M Classification - V5 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV5-Classification --model yolov5m-cls.pt --epochs 300 --batch-size 64 --imgsz 416 --name classifym_v5 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

