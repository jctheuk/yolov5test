#!/usr/bin/env python3
"""
Check Label Distribution
Check the distribution of classification labels in the dataset
"""

import os
import numpy as np
from pathlib import Path
from collections import Counter

def check_label_distribution():
    """Check the distribution of classification labels"""
    
    dataset_path = Path("Regurgitation-YOLODataset-Detection")
    train_labels_dir = dataset_path / "train" / "labels"
    
    if not train_labels_dir.exists():
        print("Train labels directory not found")
        return
    
    print("=== Checking Label Distribution ===")
    
    classifications = []
    file_class_pairs = []
    
    # Read all label files
    for file in train_labels_dir.glob('*.txt'):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                cls_line = lines[1].strip()
                classifications.append(cls_line)
                file_class_pairs.append((file.name, cls_line))
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Count classifications
    counter = Counter(classifications)
    total_files = len(classifications)
    
    print(f"\nTotal files: {total_files}")
    print(f"Classification distribution:")
    for cls, count in counter.items():
        percentage = (count / total_files) * 100
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    
    # Check first 20 files
    print(f"\nFirst 20 files:")
    for i, (filename, cls) in enumerate(file_class_pairs[:20]):
        print(f"  {i+1:2d}. {filename}: {cls}")
    
    # Check if there's a pattern
    print(f"\nChecking for patterns:")
    
    # Check if first files are all the same
    first_20_classes = [cls for _, cls in file_class_pairs[:20]]
    unique_first_20 = set(first_20_classes)
    print(f"  First 20 files unique classes: {len(unique_first_20)}")
    if len(unique_first_20) == 1:
        print(f"  ⚠️  WARNING: First 20 files are all {list(unique_first_20)[0]}")
    
    # Check if files are sorted by class
    all_classes = [cls for _, cls in file_class_pairs]
    print(f"  Total unique classes: {len(set(all_classes))}")
    
    # Find where classes change
    class_changes = []
    for i in range(1, len(all_classes)):
        if all_classes[i] != all_classes[i-1]:
            class_changes.append(i)
    
    print(f"  Class changes at positions: {class_changes[:10]}...")
    
    if len(class_changes) < 5:
        print(f"  ⚠️  WARNING: Very few class changes - files might be sorted by class!")

def check_file_sorting():
    """Check if files are sorted by class"""
    
    print(f"\n{'='*50}")
    print("CHECKING FILE SORTING")
    print(f"{'='*50}")
    
    dataset_path = Path("Regurgitation-YOLODataset-Detection")
    train_labels_dir = dataset_path / "train" / "labels"
    
    if not train_labels_dir.exists():
        return
    
    # Get all files and their classifications
    file_class_pairs = []
    
    for file in train_labels_dir.glob('*.txt'):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                cls_line = lines[1].strip()
                file_class_pairs.append((file.name, cls_line))
            
        except Exception as e:
            continue
    
    # Sort by filename
    file_class_pairs.sort(key=lambda x: x[0])
    
    print("First 30 files (sorted by filename):")
    for i, (filename, cls) in enumerate(file_class_pairs[:30]):
        print(f"  {i+1:2d}. {filename}: {cls}")
    
    # Check if sorting by filename gives better distribution
    sorted_classes = [cls for _, cls in file_class_pairs]
    unique_sorted = set(sorted_classes)
    print(f"\nUnique classes when sorted by filename: {len(unique_sorted)}")
    
    # Count consecutive same classes
    consecutive_counts = []
    current_class = sorted_classes[0]
    current_count = 1
    
    for cls in sorted_classes[1:]:
        if cls == current_class:
            current_count += 1
        else:
            consecutive_counts.append((current_class, current_count))
            current_class = cls
            current_count = 1
    
    consecutive_counts.append((current_class, current_count))
    
    print(f"Consecutive class runs:")
    for cls, count in consecutive_counts[:10]:
        print(f"  {cls}: {count} consecutive files")

def main():
    check_label_distribution()
    check_file_sorting()
    
    print(f"\n{'='*50}")
    print("CONCLUSION")
    print(f"{'='*50}")
    print("If files are sorted by class, the dataloader might be reading")
    print("consecutive files of the same class, causing the model to see")
    print("only one class during initial training epochs.")

if __name__ == "__main__":
    main()
