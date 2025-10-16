#!/bin/bash
# YOLOv5 Original Classification Training - Small Model
# Using pretrained yolov5s-cls.pt
# Batch size: 128 (optimized for classification)

# ========== YOLOv5S Classification - V1 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV1-Classification --model yolov5s-cls.pt --epochs 300 --batch-size 128 --imgsz 416 --name classifys_v1 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5S Classification - V2 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV2-Classification --model yolov5s-cls.pt --epochs 300 --batch-size 128 --imgsz 416 --name classifys_v2 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5S Classification - V3 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV3-Classification --model yolov5s-cls.pt --epochs 300 --batch-size 128 --imgsz 416 --name classifys_v3 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5S Classification - V4 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV4-Classification --model yolov5s-cls.pt --epochs 300 --batch-size 128 --imgsz 416 --name classifys_v4 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

# ========== YOLOv5S Classification - V5 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python classify/train.py --data ../regurgitationV5-Classification --model yolov5s-cls.pt --epochs 300 --batch-size 128 --imgsz 416 --name classifys_v5 --cache --nosave --optimizer Adam --lr0 0.001 --workers 4

