#!/usr/bin/env python3
"""
Script to count classification labels in the new Regurgitation dataset
"""

import os
from collections import Counter
import glob

def count_classification_labels(dataset_path):
    """Count classification labels across all splits"""
    
    splits = ['train', 'valid', 'test']
    total_counts = Counter()
    split_counts = {}
    
    print("=== Classification Label Distribution Analysis ===")
    print(f"Dataset: {dataset_path}")
    print()
    
    for split in splits:
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_path):
            print(f"❌ {split} labels directory not found: {labels_path}")
            continue
            
        # Get all .txt files
        label_files = glob.glob(os.path.join(labels_path, "*.txt"))
        
        if not label_files:
            print(f"❌ No .txt files found in {labels_path}")
            continue
            
        split_counter = Counter()
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                # Look for classification labels (one-hot encoded: 1 0 0, 0 1 0, 0 0 1)
                for line in lines:
                    line = line.strip()
                    if line in ['1 0 0', '0 1 0', '0 0 1']:
                        split_counter[line] += 1
                        total_counts[line] += 1
                        break  # Only count the first classification label per file
                        
            except Exception as e:
                print(f"❌ Error reading {label_file}: {e}")
                
        split_counts[split] = split_counter
        total_files = len(label_files)
        
        print(f"📁 {split.upper()} SET:")
        print(f"   Total files: {total_files}")
        
        if split_counter:
            print(f"   Classification distribution:")
            for label, count in sorted(split_counter.items()):
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                class_name = get_class_name(label)
                print(f"     {class_name} ({label}): {count} files ({percentage:.1f}%)")
        else:
            print(f"   ❌ No classification labels found")
        print()
    
    # Overall summary
    print("=" * 60)
    print("📊 OVERALL SUMMARY:")
    
    total_files_all = sum(len(glob.glob(os.path.join(dataset_path, split, 'labels', "*.txt"))) 
                         for split in splits 
                         if os.path.exists(os.path.join(dataset_path, split, 'labels')))
    
    print(f"   Total files across all splits: {total_files_all}")
    print(f"   Total classification labels found: {sum(total_counts.values())}")
    print()
    
    if total_counts:
        print("   Overall classification distribution:")
        for label, count in sorted(total_counts.items()):
            percentage = (count / sum(total_counts.values())) * 100 if sum(total_counts.values()) > 0 else 0
            class_name = get_class_name(label)
            print(f"     {class_name} ({label}): {count} files ({percentage:.1f}%)")
    else:
        print("   ❌ No classification labels found in any split")
    
    return total_counts, split_counts

def get_class_name(label):
    """Convert one-hot encoded label to class name"""
    if label == '1 0 0':
        return 'A4C'
    elif label == '0 1 0':
        return 'PSAX'
    elif label == '0 0 1':
        return 'PLAX'
    else:
        return f'Unknown ({label})'

if __name__ == "__main__":
    dataset_path = r"files\Regurgitation 2025_Regurgitation-YOLODataset-1-20250122T032030Z-001 (2)\Regurgitation-YOLODataset-1-20250122T032030Z-001\Regurgitation-YOLODataset-1"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        exit(1)
    
    total_counts, split_counts = count_classification_labels(dataset_path)
