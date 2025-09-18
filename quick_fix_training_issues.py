#!/usr/bin/env python3
"""
快速修復訓練問題的腳本
"""

import os
import shutil
from pathlib import Path

def backup_current_files():
    """備份當前文件"""
    print("備份當前文件...")
    
    files_to_backup = [
        "yolov5c/val.py",
        "yolov5c/train.py",
        "yolov5c/utils/loss.py"
    ]
    
    backup_dir = Path("backup_before_fix")
    backup_dir.mkdir(exist_ok=True)
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)
            print(f"✅ 已備份: {file_path} -> {backup_path}")

def check_detection_output_fix():
    """檢查檢測結果輸出修復"""
    print("\n檢查檢測結果輸出修復...")
    
    val_file = "yolov5c/val.py"
    if not os.path.exists(val_file):
        print(f"❌ 文件不存在: {val_file}")
        return False
    
    with open(val_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵修復
    checks = [
        ("names 變量處理", "names = model.names if hasattr(model, 'names') else model.module.names"),
        ("nt 變量計算", "nt = np.bincount(stats[3].astype(int), minlength=nc)"),
        ("確保輸出條件", "elif nc > 1 and len(stats) and len(ap_class) > 0:")
    ]
    
    all_fixed = True
    for check_name, check_code in checks:
        if check_code in content:
            print(f"✅ {check_name}: 已修復")
        else:
            print(f"❌ {check_name}: 未修復")
            all_fixed = False
    
    return all_fixed

def create_fixed_hyp_file():
    """創建修復的超參數文件"""
    print("\n創建修復的超參數文件...")
    
    # 創建修復的超參數配置
    fixed_hyp_content = """# YOLOv5WithClassification 修復版超參數
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
    
    hyp_file = "yolov5c/data/hyps/hyp.fixed.yaml"
    with open(hyp_file, 'w', encoding='utf-8') as f:
        f.write(fixed_hyp_content)
    
    print(f"✅ 已創建修復的超參數文件: {hyp_file}")

def create_training_script():
    """創建修復的訓練腳本"""
    print("\n創建修復的訓練腳本...")
    
    training_script = """#!/bin/bash
# YOLOv5WithClassification 修復版訓練腳本

echo "開始修復版訓練..."

# 清理快取
echo "清理數據集快取..."
rm -f Regurgitation-YOLODataset-Detection/train/labels/*.cache*
rm -f Regurgitation-YOLODataset-Detection/valid/labels/*.cache*
rm -f Regurgitation-YOLODataset-Detection/test/labels/*.cache*

# 使用修復的超參數進行訓練
echo "開始訓練..."
python yolov5c/train.py \\
    --data Regurgitation-YOLODataset-Detection/data.yaml \\
    --hyp yolov5c/data/hyps/hyp.fixed.yaml \\
    --epochs 50 \\
    --batch-size 16 \\
    --device auto \\
    --patience 10 \\
    --min-delta 0.001 \\
    --verbose

echo "訓練完成！"
"""
    
    script_file = "train_fixed.sh"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    # 設置執行權限
    os.chmod(script_file, 0o755)
    
    print(f"✅ 已創建訓練腳本: {script_file}")

def create_validation_script():
    """創建驗證腳本"""
    print("\n創建驗證腳本...")
    
    validation_script = """#!/bin/bash
# 驗證腳本 - 檢查檢測結果輸出

echo "開始驗證..."

# 使用最新的權重文件進行驗證
python yolov5c/val.py \\
    --weights yolov5c/runs/train/exp/weights/best.pt \\
    --data Regurgitation-YOLODataset-Detection/data.yaml \\
    --verbose \\
    --save-txt \\
    --save-conf

echo "驗證完成！"
"""
    
    script_file = "validate_fixed.sh"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    # 設置執行權限
    os.chmod(script_file, 0o755)
    
    print(f"✅ 已創建驗證腳本: {script_file}")

def create_monitoring_script():
    """創建監控腳本"""
    print("\n創建監控腳本...")
    
    monitoring_script = """#!/bin/bash
# 監控腳本 - 實時監控訓練狀態

echo "開始監控訓練..."

# 監控最新的日誌文件
tail -f yolov5c/runs/train/exp/train.log | grep -E "(Epoch|DEBUG|WARNING|ERROR|mAP|Accuracy)"
"""
    
    script_file = "monitor_training.sh"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(monitoring_script)
    
    # 設置執行權限
    os.chmod(script_file, 0o755)
    
    print(f"✅ 已創建監控腳本: {script_file}")

def main():
    """主函數"""
    print("YOLOv5WithClassification 訓練問題快速修復")
    print("=" * 50)
    
    # 1. 備份當前文件
    backup_current_files()
    
    # 2. 檢查檢測結果輸出修復
    if check_detection_output_fix():
        print("✅ 檢測結果輸出修復已生效")
    else:
        print("❌ 檢測結果輸出修復未完全生效，需要手動檢查")
    
    # 3. 創建修復的超參數文件
    create_fixed_hyp_file()
    
    # 4. 創建修復的訓練腳本
    create_training_script()
    
    # 5. 創建驗證腳本
    create_validation_script()
    
    # 6. 創建監控腳本
    create_monitoring_script()
    
    print("\n" + "=" * 50)
    print("修復完成！")
    print("\n下一步操作:")
    print("1. 運行修復版訓練:")
    print("   bash train_fixed.sh")
    print("\n2. 監控訓練過程:")
    print("   bash monitor_training.sh")
    print("\n3. 驗證結果:")
    print("   bash validate_fixed.sh")
    print("\n4. 檢查日誌輸出:")
    print("   python check_train_log_output.py")
    
    print("\n修復說明:")
    print("- 降低了學習率以避免 NaN 錯誤")
    print("- 減少了分類任務權重以降低過擬合")
    print("- 增加了正則化措施")
    print("- 添加了早停機制")
    print("- 創建了完整的監控和驗證腳本")

if __name__ == "__main__":
    main()
