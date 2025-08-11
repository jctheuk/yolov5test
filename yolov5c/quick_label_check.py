#!/usr/bin/env python3
"""
Quick Label Checker
Fast diagnosis of label issues
"""

import os
from pathlib import Path
import yaml

def quick_check():
    """快速檢查標註問題"""
    
    print("🔍 快速標註檢查")
    print("="*40)
    
    # 檢查數據配置
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print("❌ 數據配置文件不存在")
        return
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"📋 數據配置:")
    print(f"   檢測類別: {data_config['names']}")
    print(f"   分類類別: {data_config.get('cls_names', '未配置')}")
    
    # 檢查訓練集標註
    train_path = Path(data_config['train'])
    train_labels_path = train_path.parent / 'labels'
    
    print(f"\n📁 檢查訓練集標註: {train_labels_path}")
    
    if not train_labels_path.exists():
        print("❌ 標註目錄不存在")
        return
    
    label_files = list(train_labels_path.glob('*.txt'))
    print(f"   標註文件數量: {len(label_files)}")
    
    if not label_files:
        print("❌ 沒有標註文件")
        return
    
    # 檢查前幾個文件
    print(f"\n📄 檢查前 3 個標註文件:")
    
    detection_count = 0
    classification_count = 0
    
    for i, label_file in enumerate(label_files[:3]):
        print(f"\n   文件 {i+1}: {label_file.name}")
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"     行數: {len(lines)}")
            
            for j, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                print(f"     行 {j+1}: {line} ({len(parts)} 個值)")
                
                if len(parts) == 1:
                    classification_count += 1
                    print(f"       → 分類標註")
                elif len(parts) >= 5:
                    detection_count += 1
                    print(f"       → 檢測標註")
                else:
                    print(f"       → 未知格式")
                    
        except Exception as e:
            print(f"     ❌ 讀取錯誤: {e}")
    
    print(f"\n📊 快速統計:")
    print(f"   檢測標註: {detection_count}")
    print(f"   分類標註: {classification_count}")
    
    if classification_count == 0:
        print("\n⚠️  問題發現: 沒有分類標註!")
        print("   這會導致 YOLOv5sc 聯合訓練失敗")
        print("   建議運行: python fix_classification_labels.py")
    else:
        print("\n✅ 標註格式正常")

if __name__ == "__main__":
    quick_check()
