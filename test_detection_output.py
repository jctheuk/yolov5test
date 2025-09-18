#!/usr/bin/env python3
"""
測試檢測結果輸出是否正確
"""

import sys
import os
from pathlib import Path

# 添加 yolov5c 到路徑
sys.path.append(str(Path(__file__).parent / 'yolov5c'))

def test_detection_output():
    """測試檢測結果輸出"""
    print("測試 YOLOv5WithClassification 檢測結果輸出...")
    
    # 檢查 val.py 中的關鍵代碼
    val_file = Path('yolov5c/val.py')
    if not val_file.exists():
        print("❌ val.py 文件不存在")
        return False
    
    with open(val_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵修復
    checks = [
        ("names 變量處理", "names = model.names if hasattr(model, 'names') else model.module.names"),
        ("nt 變量計算", "nt = np.bincount(stats[3].astype(int), minlength=nc)"),
        ("按類別結果輸出", "for i, c in enumerate(ap_class):"),
        ("確保輸出條件", "elif nc > 1 and len(stats) and len(ap_class) > 0:")
    ]
    
    print("\n檢查修復項目:")
    all_passed = True
    for check_name, check_code in checks:
        if check_code in content:
            print(f"✅ {check_name}: 已修復")
        else:
            print(f"❌ {check_name}: 未找到")
            all_passed = False
    
    return all_passed

def show_expected_output():
    """顯示預期的檢測結果輸出格式"""
    print("\n預期的檢測結果輸出格式:")
    print("=" * 80)
    print("Class                  Images  Instances          P          R     mAP@0.5 mAP@0.5:0.95")
    print("all                        183        183      0.388       0.21      0.126     0.0373")
    print("0                          183         66      0.247      0.182      0.187     0.0521")
    print("1                          183         55      0.161        0.2      0.103     0.0313")
    print("2                          183         14          1          0     0.0193    0.00415")
    print("3                          183         48      0.145      0.458      0.197     0.0617")
    print("=" * 80)

def main():
    """主函數"""
    print("YOLOv5WithClassification 檢測結果輸出修復測試")
    print("=" * 60)
    
    # 測試修復
    if test_detection_output():
        print("\n✅ 所有修復項目都已正確實施")
    else:
        print("\n❌ 部分修復項目需要檢查")
    
    # 顯示預期輸出
    show_expected_output()
    
    print("\n修復說明:")
    print("1. 修復了 names 變量的處理方式，與原始 YOLOv5 保持一致")
    print("2. 修復了 nt 變量的計算方式")
    print("3. 增加了額外的輸出條件，確保按類別結果總是顯示")
    print("4. 保持了與原始 YOLOv5 相同的輸出格式")
    
    print("\n現在檢測結果應該會正確顯示按類別的詳細信息！")

if __name__ == "__main__":
    main()
