#!/usr/bin/env python3
"""
Remove duplicate classification labels from label files
Remove lines containing square brackets (4th line) to keep only the 2nd line classification
"""

import os
import glob
from pathlib import Path

def remove_duplicate_labels(data_dir):
    """
    Remove duplicate classification labels from label files
    
    Keep:
    - Line 1: Detection labels
    - Line 2: Classification labels (one-hot encoding)
    - Line 3: Empty line
    
    Remove:
    - Line 4: Duplicate classification labels with square brackets
    - Line 5: Empty line
    """
    
    # Find all label files
    label_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"Found {len(label_files)} label files in {data_dir}")
    
    fixed_count = 0
    error_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Check if file has the problematic format (5 lines with brackets on line 4)
            if len(lines) >= 4:
                line4 = lines[3].strip()  # 4th line (index 3)
                
                # Check if line 4 contains square brackets
                if '[' in line4 and ']' in line4:
                    # Keep only first 3 lines
                    new_lines = lines[:3]
                    
                    # Write back to file
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    
                    fixed_count += 1
                    print(f"Fixed {os.path.basename(label_file)}: Removed duplicate label {line4}")
                    
                elif len(lines) > 3:
                    # File has more than 3 lines but no brackets on line 4
                    print(f"Warning: {os.path.basename(label_file)} has {len(lines)} lines but no brackets on line 4")
                    
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            error_count += 1
    
    print(f"\nSummary for {data_dir}:")
    print(f"Fixed: {fixed_count} files")
    print(f"Errors: {error_count} files")
    print(f"Total: {len(label_files)} files")
    
    return fixed_count, error_count

def main():
    """Main function to remove duplicate labels from all dataset splits"""
    print("=== 移除重複分類標籤 ===")
    
    dataset_path = "Regurgitation-YOLODataset-Detection"
    total_fixed = 0
    total_errors = 0
    
    # Process each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        labels_dir = os.path.join(dataset_path, split, "labels")
        if os.path.exists(labels_dir):
            print(f"\n--- 處理 {split} 集 ---")
            fixed, errors = remove_duplicate_labels(labels_dir)
            total_fixed += fixed
            total_errors += errors
        else:
            print(f"\n--- {split} 集不存在: {labels_dir} ---")
    
    print(f"\n=== 總計 ===")
    print(f"總共修復: {total_fixed} 個文件")
    print(f"總共錯誤: {total_errors} 個文件")
    
    if total_fixed > 0:
        print(f"\n✅ 成功移除重複標籤！")
        print(f"記得清除快取文件:")
        print(f"Remove-Item -Path \"{dataset_path}/*/labels/*.cache*\" -Force -ErrorAction SilentlyContinue")
    else:
        print(f"\nℹ️ 沒有找到需要修復的文件")

if __name__ == "__main__":
    main()
