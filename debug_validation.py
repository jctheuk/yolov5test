#!/usr/bin/env python3
import sys
import os
sys.path.append('yolov5c')

from yolov5c.val import run
from yolov5c.utils.general import LOGGER
import torch

def debug_validation():
    # 設置調試模式
    LOGGER.setLevel(10)  # DEBUG level
    
    # 運行驗證並捕獲詳細信息
    try:
        results = run(
            weights='files/testingclassificationv26/weights/best.pt',
            data='Regurgitation-YOLODataset-Detection/data.yaml',
            batch_size=16,
            imgsz=416,
            conf_thres=0.001,
            iou_thres=0.6,
            task='val',
            device='',
            single_cls=False,
            augment=False,
            verbose=True,
            save_txt=False,
            save_hybrid=False,
            save_conf=False,
            save_json=False,
            project='yolov5c/runs/val',
            name='exp',
            exist_ok=False,
            half=False,
            dnn=False
        )
        print("驗證完成，結果:", results)
    except Exception as e:
        print(f"驗證過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_validation()
