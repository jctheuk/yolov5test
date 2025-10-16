#!/bin/bash
# YOLOv5 Original Classification Training - Small Model
# Using pretrained yolov5s-cls.pt

# ========== YOLOv5S Classification - V1 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV1_classify --model yolov5s-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifys_v1 --cache --nosave

# ========== YOLOv5S Classification - V2 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV2_classify --model yolov5s-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifys_v2 --cache --nosave

# ========== YOLOv5S Classification - V3 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV3_classify --model yolov5s-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifys_v3 --cache --nosave

# ========== YOLOv5S Classification - V4 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV4_classify --model yolov5s-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifys_v4 --cache --nosave

# ========== YOLOv5S Classification - V5 ==========
cd /work/jonchang3909/yolov5test/yolov5original/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas && sudo pip install seaborn && sudo pip install scikit-learn && python classify/train.py --data ../regurgitationV5_classify --model yolov5s-cls.pt --epochs 300 --batch-size 128 --img 416 --name classifys_v5 --cache --nosave
