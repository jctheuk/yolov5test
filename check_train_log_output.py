#!/usr/bin/env python3
"""
檢查訓練日誌文件輸出是否符合預期
"""

import os
import re
import glob
from pathlib import Path
import pandas as pd

def find_latest_log_files():
    """找到最新的日誌文件"""
    log_patterns = [
        "*.log",
        "files/*.log", 
        "files/job_*.log",
        "runs/train/*/train.log",
        "yolov5c/runs/train/*/train.log"
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    # 按修改時間排序，最新的在前
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    return log_files

def analyze_log_output(log_file):
    """分析日誌文件輸出"""
    print(f"\n分析日誌文件: {log_file}")
    print("=" * 60)
    
    if not os.path.exists(log_file):
        print(f"❌ 日誌文件不存在: {log_file}")
        return False
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 檢查關鍵輸出項目
    checks = {
        "檢測結果按類別輸出": {
            "pattern": r"^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+$",
            "description": "檢測結果的按類別詳細輸出"
        },
        "分類結果輸出": {
            "pattern": r"Classification Results:",
            "description": "分類任務結果輸出"
        },
        "DEBUG 輸出": {
            "pattern": r"\[DEBUG\]",
            "description": "調試信息輸出"
        },
        "過擬合警告": {
            "pattern": r"\[DEBUG\] WARNING: Model is predicting only class",
            "description": "過擬合檢測警告"
        },
        "訓練進度": {
            "pattern": r"Epoch.*GPU_mem.*box_loss.*obj_loss.*cls_loss.*cls_task_loss",
            "description": "訓練進度輸出"
        },
        "驗證結果": {
            "pattern": r"Class.*Images.*Instances.*P.*R.*mAP50.*mAP50-95",
            "description": "驗證結果標題"
        }
    }
    
    results = {}
    for check_name, check_info in checks.items():
        matches = re.findall(check_info["pattern"], content, re.MULTILINE)
        results[check_name] = {
            "found": len(matches) > 0,
            "count": len(matches),
            "description": check_info["description"]
        }
    
    # 顯示檢查結果
    print("檢查結果:")
    for check_name, result in results.items():
        status = "✅" if result["found"] else "❌"
        print(f"{status} {check_name}: {result['count']} 次 - {result['description']}")
    
    return results

def extract_detection_results(log_file):
    """提取檢測結果"""
    print(f"\n提取檢測結果:")
    print("-" * 40)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找檢測結果表格
    detection_pattern = r"(Class.*Images.*Instances.*P.*R.*mAP50.*mAP50-95.*\n(?:.*\n)*)"
    matches = re.findall(detection_pattern, content, re.MULTILINE)
    
    if matches:
        print("找到檢測結果:")
        for i, match in enumerate(matches[-3:]):  # 顯示最後3次
            print(f"\n第 {i+1} 次檢測結果:")
            print(match.strip())
    else:
        print("❌ 未找到檢測結果表格")
    
    # 查找按類別的詳細結果
    class_pattern = r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$"
    class_matches = re.findall(class_pattern, content, re.MULTILINE)
    
    if class_matches:
        print(f"\n找到 {len(class_matches)} 行按類別結果")
        print("最新幾行:")
        for match in class_matches[-5:]:  # 顯示最後5行
            class_id, images, instances, p, r, map50, map = match
            print(f"  類別 {class_id}: Images={images}, Instances={instances}, P={p}, R={r}, mAP50={map50}, mAP={map}")
    else:
        print("❌ 未找到按類別的詳細結果")

def extract_classification_results(log_file):
    """提取分類結果"""
    print(f"\n提取分類結果:")
    print("-" * 40)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找分類結果
    cls_pattern = r"Classification Results - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)"
    cls_matches = re.findall(cls_pattern, content)
    
    if cls_matches:
        print(f"找到 {len(cls_matches)} 次分類結果")
        print("最新幾次:")
        for i, match in enumerate(cls_matches[-3:]):  # 顯示最後3次
            acc, prec, rec, f1 = match
            print(f"  第 {i+1} 次: Accuracy={acc}, Precision={prec}, Recall={rec}, F1={f1}")
    else:
        print("❌ 未找到分類結果")
    
    # 查找過擬合警告
    overfitting_pattern = r"\[DEBUG\] WARNING: Model is predicting only class (\d+) \(overfitting\)"
    overfitting_matches = re.findall(overfitting_pattern, content)
    
    if overfitting_matches:
        print(f"\n過擬合警告: {len(overfitting_matches)} 次")
        class_counts = {}
        for class_id in overfitting_matches:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        print("預測類別分布:")
        for class_id, count in class_counts.items():
            print(f"  類別 {class_id}: {count} 次")
    else:
        print("✅ 無過擬合警告")

def check_training_progress(log_file):
    """檢查訓練進度"""
    print(f"\n檢查訓練進度:")
    print("-" * 40)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找訓練進度
    progress_pattern = r"Epoch\s+(\d+).*GPU_mem.*box_loss.*obj_loss.*cls_loss.*cls_task_loss.*Instances.*Size"
    progress_matches = re.findall(progress_pattern, content)
    
    if progress_matches:
        print(f"找到 {len(progress_matches)} 個訓練輪數")
        print(f"最新輪數: {progress_matches[-1]}")
        
        # 查找損失值
        loss_pattern = r"all\s+(\d+)\s+(\d+)\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)"
        loss_matches = re.findall(loss_pattern, content)
        
        if loss_matches:
            print(f"找到 {len(loss_matches)} 行損失數據")
            print("最新損失值:")
            for match in loss_matches[-3:]:  # 顯示最後3行
                images, instances, box_loss, obj_loss, cls_loss, cls_task_loss = match
                print(f"  Images={images}, Instances={instances}")
                print(f"  Box Loss={box_loss}, Obj Loss={obj_loss}, Cls Loss={cls_loss}, Cls Task Loss={cls_task_loss}")
    else:
        print("❌ 未找到訓練進度")

def generate_summary_report(log_files, results):
    """生成摘要報告"""
    print(f"\n" + "=" * 60)
    print("訓練日誌輸出檢查摘要報告")
    print("=" * 60)
    
    print(f"檢查的日誌文件數量: {len(log_files)}")
    print(f"最新日誌文件: {log_files[0] if log_files else '無'}")
    
    print(f"\n功能檢查結果:")
    all_good = True
    for check_name, result in results.items():
        status = "✅ 正常" if result["found"] else "❌ 缺失"
        print(f"  {check_name}: {status} ({result['count']} 次)")
        if not result["found"]:
            all_good = False
    
    print(f"\n總體評估:")
    if all_good:
        print("✅ 所有預期輸出都正常")
    else:
        print("⚠️ 部分輸出缺失，需要檢查")
    
    print(f"\n建議:")
    if not results.get("檢測結果按類別輸出", {}).get("found", False):
        print("- 檢測結果按類別輸出缺失，可能需要重新運行驗證")
    if not results.get("分類結果輸出", {}).get("found", False):
        print("- 分類結果輸出缺失，檢查分類功能是否啟用")
    if results.get("過擬合警告", {}).get("count", 0) > 0:
        print("- 發現過擬合警告，建議調整訓練參數")

def main():
    """主函數"""
    print("YOLOv5WithClassification 訓練日誌輸出檢查")
    print("=" * 60)
    
    # 找到最新的日誌文件
    log_files = find_latest_log_files()
    
    if not log_files:
        print("❌ 未找到任何日誌文件")
        print("請確保以下位置有日誌文件:")
        print("  - *.log")
        print("  - files/*.log")
        print("  - runs/train/*/train.log")
        return
    
    print(f"找到 {len(log_files)} 個日誌文件:")
    for i, log_file in enumerate(log_files[:5]):  # 顯示前5個
        print(f"  {i+1}. {log_file}")
    
    # 分析最新的日誌文件
    latest_log = log_files[0]
    results = analyze_log_output(latest_log)
    
    # 提取詳細結果
    extract_detection_results(latest_log)
    extract_classification_results(latest_log)
    check_training_progress(latest_log)
    
    # 生成摘要報告
    generate_summary_report(log_files, results)
    
    print(f"\n檢查完成！")
    print(f"如需檢查其他日誌文件，請運行:")
    print(f"python check_train_log_output.py")

if __name__ == "__main__":
    main()
