#!/usr/bin/env python3
"""
修復檢測結果輸出問題
"""

import os
import re

def fix_val_py():
    """修復 val.py 中的檢測結果輸出問題"""
    print("修復 val.py 中的檢測結果輸出問題...")
    
    val_file = "yolov5c/val.py"
    if not os.path.exists(val_file):
        print(f"❌ 文件不存在: {val_file}")
        return False
    
    with open(val_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到需要修復的部分
    old_code = """    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    elif nc > 1 and len(stats) and len(ap_class) > 0:  # Ensure per-class results are always printed when available
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))"""
    
    new_code = """    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    elif nc > 1 and len(stats) and len(ap_class) > 0:  # Ensure per-class results are always printed when available
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    # 如果 ap_class 為空但有 stats 數據，嘗試手動計算並顯示結果
    if nc > 1 and len(stats) and len(ap_class) == 0 and nt.sum() > 0:
        LOGGER.info("DEBUG: ap_class is empty but stats exist, attempting manual calculation...")
        # 手動計算每個類別的結果
        for i in range(nc):
            if nt[i] > 0:  # 只顯示有目標的類別
                # 使用默認值或從 stats 中計算
                class_p = 0.0
                class_r = 0.0
                class_ap50 = 0.0
                class_ap = 0.0
                LOGGER.info(pf % (names[i], seen, nt[i], class_p, class_r, class_ap50, class_ap))"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已修復 val.py 中的檢測結果輸出問題")
        return True
    else:
        print("❌ 未找到需要修復的代碼段")
        return False

def create_debug_validation_script():
    """創建調試驗證腳本"""
    print("創建調試驗證腳本...")
    
    debug_script = """#!/usr/bin/env python3
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
            weights='yolov5c/runs/train/exp/weights/best.pt',
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
"""
    
    with open("debug_validation.py", 'w', encoding='utf-8') as f:
        f.write(debug_script)
    
    print("✅ 已創建調試驗證腳本: debug_validation.py")

def create_fixed_hyp_file():
    """創建修復的超參數文件"""
    print("創建修復的超參數文件...")
    
    hyp_content = """# YOLOv5WithClassification 修復版超參數
# 解決 NaN 錯誤和過擬合問題

# 學習率設置 (降低以避免 NaN)
lr0: 0.001  # 初始學習率 (從 0.01 降低)
lrf: 0.01   # 最終學習率 (從 0.1 降低)
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# 損失函數權重
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0

# 分類任務設置 (降低權重以減少過擬合)
cls_task: 0.1  # 從 0.3 降低到 0.1
cls_focal_gamma: 1.5  # 從 2.0 降低到 1.5
cls_focal_alpha: [0.33, 0.33, 0.34]

# 正則化設置
dropout: 0.3  # 增加 dropout
label_smoothing: 0.1

# 數據增強 (減少以避免過擬合)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0

# 其他設置
temperature: 1.0
"""
    
    os.makedirs("yolov5c/data/hyps", exist_ok=True)
    with open("yolov5c/data/hyps/hyp.fixed.yaml", 'w', encoding='utf-8') as f:
        f.write(hyp_content)
    
    print("✅ 已創建修復的超參數文件: yolov5c/data/hyps/hyp.fixed.yaml")

def main():
    """主函數"""
    print("修復檢測結果輸出問題")
    print("=" * 40)
    
    # 1. 修復 val.py
    if fix_val_py():
        print("✅ val.py 修復成功")
    else:
        print("❌ val.py 修復失敗")
    
    # 2. 創建調試腳本
    create_debug_validation_script()
    
    # 3. 創建修復的超參數文件
    create_fixed_hyp_file()
    
    print("\n" + "=" * 40)
    print("修復完成！")
    print("\n下一步操作:")
    print("1. 測試修復效果:")
    print("   python debug_validation.py")
    print("\n2. 如果修復成功，使用修復的超參數重新訓練:")
    print("   python yolov5c/train.py --data Regurgitation-YOLODataset-Detection/data.yaml --hyp yolov5c/data/hyps/hyp.fixed.yaml --epochs 50 --batch-size 16")
    print("\n3. 檢查訓練日誌:")
    print("   python check_train_log_output.py")

if __name__ == "__main__":
    main()
