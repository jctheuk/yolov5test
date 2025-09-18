#!/usr/bin/env python3
"""
Analyze classification class ratios in regurgitationBalanced dataset
"""

import os
from collections import Counter

def analyze_classification_ratio():
    # Classification class names
    cls_names = ['A4C', 'PSAX', 'PLAX']
    
    print("=== regurgitationBalanced Classification Class Analysis ===")
    print()
    
    # Count classification labels in each split
    for split in ['train', 'valid', 'test']:
        labels_dir = f'regurgitationBalanced/{split}/labels'
        if not os.path.exists(labels_dir):
            print(f"{split}: directory not found")
            continue
            
        print(f"=== {split.upper()} SET ===")
        
        # Count classification class occurrences
        cls_counts = Counter()
        total_files = 0
        
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                total_files += 1
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 6:  # YOLO format with classification: class x y w h cls
                                    cls_id = int(parts[5])  # Classification class is 6th element
                                    if 0 <= cls_id < len(cls_names):
                                        cls_counts[cls_id] += 1
                except Exception as e:
                    continue
        
        print(f"Total files: {total_files}")
        print("Classification class distribution:")
        for cls_id in range(len(cls_names)):
            count = cls_counts[cls_id]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"  {cls_names[cls_id]}: {count} files ({percentage:.1f}%)")
        print()
    
    # Overall statistics
    print("=== OVERALL STATISTICS ===")
    total_cls_counts = Counter()
    total_files = 0
    
    for split in ['train', 'valid', 'test']:
        labels_dir = f'regurgitationBalanced/{split}/labels'
        if os.path.exists(labels_dir):
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(labels_dir, label_file)
                    total_files += 1
                    
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 6:
                                        cls_id = int(parts[5])
                                        if 0 <= cls_id < len(cls_names):
                                            total_cls_counts[cls_id] += 1
                    except:
                        continue
    
    print(f"Total files across all splits: {total_files}")
    print("Overall classification class distribution:")
    for cls_id in range(len(cls_names)):
        count = total_cls_counts[cls_id]
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"  {cls_names[cls_id]}: {count} files ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_classification_ratio()



