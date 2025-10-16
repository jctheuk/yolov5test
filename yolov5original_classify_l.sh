#!/bin/bash
# YOLOv5 Original Classification Training - Large Model
# Using pretrained yolov5l-cls.pt

# ========== YOLOv5L Classification - V1 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV1_classify --model yolov5l-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifyl_v1 --cache --nosave

# ========== YOLOv5L Classification - V2 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && python classify/train.py --data ../regurgitationV2_classify --model yolov5l-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifyl_v2 --cache --nosave

# ========== YOLOv5L Classification - V3 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && python classify/train.py --data ../regurgitationV3_classify --model yolov5l-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifyl_v3 --cache --nosave

# ========== YOLOv5L Classification - V4 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && python classify/train.py --data ../regurgitationV4_classify --model yolov5l-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifyl_v4 --cache --nosave

# ========== YOLOv5L Classification - V5 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && python classify/train.py --data ../regurgitationV5_classify --model yolov5l-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifyl_v5 --cache --nosave
