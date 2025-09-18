#!/usr/bin/env python3
"""
快速驗證測試 - 只測試少量數據
"""

import sys
import os
sys.path.append('yolov5c')

def quick_test():
    """快速測試修復效果"""
    print("快速驗證測試")
    print("=" * 40)
    
    try:
        from yolov5c.val import run
        from yolov5c.utils.general import LOGGER
        
        # 設置較小的批次大小和圖像數量
        print("開始快速驗證...")
        
        results = run(
            weights='files/testingclassificationv26/weights/best.pt',
            data='Regurgitation-YOLODataset-Detection/data.yaml',
            batch_size=4,  # 小批次
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
            project='quick_test',
            name='exp',
            exist_ok=True,
            half=False,
            dnn=False
        )
        
        print("\n快速驗證完成！")
        print("結果:", results)
        
    except Exception as e:
        print(f"測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
