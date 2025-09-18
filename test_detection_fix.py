#!/usr/bin/env python3
"""
測試檢測結果輸出修復效果
"""

import sys
import os
sys.path.append('yolov5c')

from yolov5c.val import run
from yolov5c.utils.general import LOGGER
import torch

def test_detection_output_fix():
    """測試檢測結果輸出修復"""
    print("測試檢測結果輸出修復效果")
    print("=" * 50)
    
    # 設置詳細日誌
    LOGGER.setLevel(20)  # INFO level
    
    try:
        print("使用權重文件: files/testingclassificationv26/weights/best.pt")
        print("使用數據集: Regurgitation-YOLODataset-Detection/data.yaml")
        print("\n開始驗證...")
        
        # 運行驗證
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
            verbose=True,  # 啟用詳細輸出
            save_txt=False,
            save_hybrid=False,
            save_conf=False,
            save_json=False,
            project='test_detection_fix',
            name='exp',
            exist_ok=True,
            half=False,
            dnn=False
        )
        
        print("\n" + "=" * 50)
        print("驗證完成！")
        print("結果:", results)
        
        # 檢查是否有按類別的檢測結果輸出
        print("\n檢查修復效果:")
        print("如果看到類似以下的輸出，說明修復成功:")
        print("Class     Images  Instances          P          R      mAP50   mAP50-95")
        print("all        181        181   0.000515      0.145    0.00123   0.000173")
        print("0          181         66      0.247      0.182      0.187     0.0521")
        print("1          181         55      0.161        0.2      0.103     0.0313")
        print("2          181         14          1          0     0.0193    0.00415")
        print("3          181         48      0.145      0.458      0.197     0.0617")
        
    except Exception as e:
        print(f"驗證過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_output_fix()
