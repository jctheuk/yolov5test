#!/usr/bin/env python3
"""
簡單測試腳本
"""

import sys
import os

print("開始測試...")

# 檢查文件是否存在
weight_file = "files/testingclassificationv26/weights/best.pt"
data_file = "Regurgitation-YOLODataset-Detection/data.yaml"

print(f"檢查權重文件: {weight_file}")
print(f"存在: {os.path.exists(weight_file)}")

print(f"檢查數據文件: {data_file}")
print(f"存在: {os.path.exists(data_file)}")

# 嘗試導入模塊
try:
    sys.path.append('yolov5c')
    print("添加 yolov5c 到路徑")
    
    from yolov5c.val import run
    print("成功導入 run 函數")
    
    from yolov5c.utils.general import LOGGER
    print("成功導入 LOGGER")
    
    import torch
    print("成功導入 torch")
    
    print("所有模塊導入成功！")
    
except Exception as e:
    print(f"導入錯誤: {e}")
    import traceback
    traceback.print_exc()

print("測試完成")
