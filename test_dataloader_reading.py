#!/usr/bin/env python3
"""
Test script to check what the dataloader actually reads from label files
"""

import sys
import os
sys.path.append('yolov5c')

from utils.dataloaders import verify_image_label
import numpy as np

def test_dataloader_reading():
    """Test what the dataloader actually reads"""
    print("=== 數據加載器讀取測試 ===")
    
    # Test files with different formats
    test_files = [
        "Regurgitation-YOLODataset-Detection/train/labels/ZmZqwqxoa8KZwps=-unnamed_1_1.mp4-7.txt",
        "Regurgitation-YOLODataset-Detection/train/labels/ZmZnwqlqbMKawp0=-unnamed_2_8.mp4-7.txt",
        "Regurgitation-YOLODataset-Detection/valid/labels/ZmZqwqxoa8KZwps=-unnamed_1_1.mp4-48.txt"
    ]
    
    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"\n--- 測試文件: {os.path.basename(filepath)} ---")
        
        # Read raw file content
        print("原始文件內容:")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                print(f"  第{i+1}行: {repr(line.strip())}")
        
        # Test what dataloader reads
        print("\n數據加載器解析結果:")
        try:
            # Simulate dataloader reading
            detection_lines = []
            classification_line = None
            
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                parts = line.split()
                if len(parts) == 5:  # Detection line
                    detection_lines.append(parts)
                    print(f"  檢測標籤: {parts}")
                elif len(parts) == 3:  # Classification line
                    classification_line = parts
                    print(f"  分類標籤: {parts}")
                else:
                    print(f"  跳過格式錯誤行: {parts}")
            
            # Process classification line like dataloader does
            print(f"\n分類標籤處理:")
            if classification_line is not None:
                try:
                    cls_values = [float(x) for x in classification_line]
                    if len(cls_values) == 3:
                        print(f"  原始: {cls_values}")
                        print(f"  最終: {cls_values}")
                        
                        # Check which class this represents
                        class_idx = np.argmax(cls_values)
                        class_names = ['PSAX', 'PLAX', 'A4C']
                        print(f"  對應類別: {class_names[class_idx]} (索引 {class_idx})")
                    else:
                        print(f"  格式錯誤: {cls_values}")
                except (ValueError, IndexError) as e:
                    print(f"  解析錯誤: {e}")
                    print(f"  使用默認值: [1.0, 0.0, 0.0]")
            else:
                print(f"  沒有找到分類標籤，使用默認值: [1.0, 0.0, 0.0]")
                
        except Exception as e:
            print(f"  處理錯誤: {e}")

def test_multiple_files():
    """Test multiple files to see patterns"""
    print("\n=== 多文件模式分析 ===")
    
    # Check a few files from each video ID
    video_patterns = {
        'ZmZqwqxoa8KZwps=-unnamed_1_1.mp4': [],
        'ZmZnwqlqbMKawp0=-unnamed_2_8.mp4': [],
        'ZmZiwqZua8KawqA=-unnamed_1_2.mp4': []
    }
    
    label_dir = "Regurgitation-YOLODataset-Detection/train/labels"
    
    for filename in os.listdir(label_dir):
        for video_id in video_patterns.keys():
            if video_id in filename:
                filepath = os.path.join(label_dir, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        # Get classification line (second line)
                        cls_line = lines[1].strip()
                        if cls_line:
                            parts = cls_line.split()
                            if len(parts) == 3:
                                video_patterns[video_id].append(parts)
                break
    
    print("各視頻ID的分類標籤模式:")
    for video_id, labels in video_patterns.items():
        if labels:
            print(f"\n{video_id}:")
            unique_labels = set(tuple(label) for label in labels)
            for label in unique_labels:
                count = labels.count(list(label))
                class_idx = np.argmax([float(x) for x in label])
                class_names = ['PSAX', 'PLAX', 'A4C']
                print(f"  {label}: {class_names[class_idx]} ({count} 次)")

if __name__ == "__main__":
    test_dataloader_reading()
    test_multiple_files()
    
    print("\n=== 結論 ===")
    print("數據加載器讀取邏輯:")
    print("1. 讀取所有非空行")
    print("2. 5個數字的行 = 檢測標籤")
    print("3. 3個數字的行 = 分類標籤 (只取第一個遇到的)")
    print("4. 跳過其他格式的行")
    print("5. 如果沒有分類標籤，使用默認值 [1.0, 0.0, 0.0]")
