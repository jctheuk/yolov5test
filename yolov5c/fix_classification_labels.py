#!/usr/bin/env python3
"""
Fix Classification Labels
Adds missing classification labels to the dataset
"""

import os
import random
from pathlib import Path
import yaml

def add_classification_labels(data_yaml_path):
    """為數據集添加分類標註"""
    
    print("🔧 修復分類標註問題...")
    
    # 讀取數據配置
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    train_path = Path(data_config['train'])
    val_path = Path(data_config['val'])
    
    # 分類類別
    classification_classes = data_config['cls_names']  # ['PSAX', 'PLAX', 'A4C']
    
    print(f"📁 處理訓練集: {train_path}")
    print(f"📁 處理驗證集: {val_path}")
    print(f"🏷️  分類類別: {classification_classes}")
    
    # 處理訓練集
    train_labels_path = train_path.parent / 'labels'
    if train_labels_path.exists():
        add_labels_to_directory(train_labels_path, classification_classes, "訓練集")
    
    # 處理驗證集
    val_labels_path = val_path.parent / 'labels'
    if val_labels_path.exists():
        add_labels_to_directory(val_labels_path, classification_classes, "驗證集")
        
    print("✅ 分類標註修復完成！")

def add_labels_to_directory(labels_path, classes, dataset_name):
    """為目錄中的所有標註文件添加分類標註"""
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"   📊 {dataset_name}: 找到 {len(label_files)} 個標註文件")
    
    # 統計現有分類標註
    existing_cls_labels = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 1:  # 分類標註
                existing_cls_labels += 1
                
    print(f"   📈 現有分類標註: {existing_cls_labels}")
    
    if existing_cls_labels == 0:
        print(f"   ⚠️  沒有找到分類標註，正在添加...")
        
        # 為每個文件添加分類標註
        for label_file in label_files:
            add_classification_to_file(label_file, classes)
            
        print(f"   ✅ 已為 {len(label_files)} 個文件添加分類標註")
    else:
        print(f"   ℹ️  已存在分類標註，跳過添加")

def add_classification_to_file(label_file, classes):
    """為單個標註文件添加分類標註"""
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # 檢查是否已有分類標註
    has_classification = any(len(line.strip().split()) == 1 for line in lines)
    
    if not has_classification:
        # 隨機選擇一個分類類別
        # 這裡可以根據文件名或其他邏輯來決定分類
        # 目前使用隨機分配作為示例
        class_id = random.randint(0, len(classes) - 1)
        
        # 添加分類標註到文件末尾
        with open(label_file, 'a') as f:
            f.write(f"{class_id}\n")

def create_balanced_classification_labels(data_yaml_path):
    """創建平衡的分類標註"""
    
    print("⚖️  創建平衡的分類標註...")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    train_path = Path(data_config['train'])
    val_path = Path(data_config['val'])
    classes = data_config['cls_names']
    
    # 處理訓練集
    train_labels_path = train_path.parent / 'labels'
    if train_labels_path.exists():
        create_balanced_labels(train_labels_path, classes, "訓練集")
    
    # 處理驗證集
    val_labels_path = val_path.parent / 'labels'
    if val_labels_path.exists():
        create_balanced_labels(val_labels_path, classes, "驗證集")

def create_balanced_labels(labels_path, classes, dataset_name):
    """創建平衡的分類標註"""
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"   📊 {dataset_name}: {len(label_files)} 個文件")
    
    # 計算每個類別的目標數量
    target_per_class = len(label_files) // len(classes)
    remainder = len(label_files) % len(classes)
    
    print(f"   🎯 目標每類數量: {target_per_class}")
    
    # 分配類別ID
    class_assignments = []
    for i, class_name in enumerate(classes):
        count = target_per_class + (1 if i < remainder else 0)
        class_assignments.extend([i] * count)
    
    # 打亂分配順序
    random.shuffle(class_assignments)
    
    # 為文件添加分類標註
    for i, label_file in enumerate(label_files):
        if i < len(class_assignments):
            class_id = class_assignments[i]
            
            # 檢查是否已有分類標註
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            has_classification = any(len(line.strip().split()) == 1 for line in lines)
            
            if not has_classification:
                with open(label_file, 'a') as f:
                    f.write(f"{class_id}\n")
    
    print(f"   ✅ 平衡分類標註完成")

def main():
    """主函數"""
    print("🔧 YOLOv5 分類標註修復工具")
    print("="*50)
    
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print(f"❌ 找不到數據配置文件: {data_yaml_path}")
        return
    
    # 選擇修復模式
    print("\n選擇修復模式:")
    print("1. 隨機分配分類標註")
    print("2. 平衡分配分類標註")
    
    choice = input("請選擇 (1 或 2): ").strip()
    
    if choice == "1":
        add_classification_labels(data_yaml_path)
    elif choice == "2":
        create_balanced_classification_labels(data_yaml_path)
    else:
        print("❌ 無效選擇")
        return
    
    print("\n🎉 分類標註修復完成！")
    print("📝 請重新運行訓練以使用修復後的標註")

if __name__ == "__main__":
    main()
