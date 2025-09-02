#!/usr/bin/env python3
"""
Analysis script to identify classification performance issues
Analyzes label distribution, data quality, and training configuration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from collections import Counter
import ast

def analyze_classification_labels(dataset_path):
    """Analyze classification label distribution and quality"""
    print(f"Analyzing classification labels in: {dataset_path}")
    
    # Load data.yaml
    data_yaml_path = Path(dataset_path) / "data.yaml"
    if not data_yaml_path.exists():
        print(f"ERROR: data.yaml not found at {data_yaml_path}")
        return
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset configuration:")
    print(f"  Detection classes: {data_config.get('names', [])}")
    print(f"  Classification classes: {data_config.get('cls_names', [])}")
    print(f"  Number of detection classes: {data_config.get('nc', 0)}")
    print(f"  Number of classification classes: {data_config.get('num_cls', 0)}")
    
    # Analyze each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            print(f"WARNING: {split} directory not found")
            continue
            
        labels_path = split_path / "labels"
        if not labels_path.exists():
            print(f"WARNING: {split}/labels directory not found")
            continue
        
        print(f"\n=== {split.upper()} SET ANALYSIS ===")
        analyze_split_labels(labels_path, data_config)

def analyze_split_labels(labels_path, data_config):
    """Analyze labels in a specific split"""
    label_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("No label files found!")
        return
    
    # Statistics
    detection_labels = []
    classification_labels = []
    files_with_classification = 0
    files_with_detection = 0
    total_files = len(label_files)
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        detection_found = False
        classification_found = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # Classification label
                try:
                    cls_label = ast.literal_eval(line)
                    classification_labels.append(cls_label)
                    classification_found = True
                except:
                    print(f"ERROR: Invalid classification label in {label_file}: {line}")
            elif line and not line.startswith('#'):
                # Detection label
                try:
                    parts = line.split()
                    if len(parts) >= 5:
                        detection_labels.append([float(x) for x in parts])
                        detection_found = True
                except:
                    print(f"ERROR: Invalid detection label in {label_file}: {line}")
        
        if classification_found:
            files_with_classification += 1
        if detection_found:
            files_with_detection += 1
    
    print(f"Files with classification labels: {files_with_classification}/{total_files} ({files_with_classification/total_files*100:.1f}%)")
    print(f"Files with detection labels: {files_with_detection}/{total_files} ({files_with_detection/total_files*100:.1f}%)")
    
    # Analyze classification distribution
    if classification_labels:
        analyze_classification_distribution(classification_labels, data_config)
    
    # Analyze detection distribution
    if detection_labels:
        analyze_detection_distribution(detection_labels, data_config)

def analyze_classification_distribution(classification_labels, data_config):
    """Analyze classification label distribution"""
    print("\n--- Classification Label Analysis ---")
    
    # Convert to class indices
    class_indices = []
    for label in classification_labels:
        if isinstance(label, list):
            if len(label) == 3:  # One-hot encoding
                class_idx = np.argmax(label)
                class_indices.append(class_idx)
            else:
                class_idx = int(label[0]) if label else 0
                class_indices.append(class_idx)
        else:
            class_indices.append(int(label))
    
    # Count distribution
    class_counts = Counter(class_indices)
    total_samples = len(class_indices)
    
    print(f"Total classification samples: {total_samples}")
    print("Class distribution:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = count / total_samples * 100
        class_name = data_config.get('cls_names', [])[class_idx] if class_idx < len(data_config.get('cls_names', [])) else f"Class_{class_idx}"
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Check for imbalance
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("WARNING: Significant class imbalance detected!")
        elif imbalance_ratio > 1.5:
            print("NOTE: Moderate class imbalance detected")
        else:
            print("Class distribution appears balanced")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    class_names = data_config.get('cls_names', [f"Class_{i}" for i in range(len(class_counts))])
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    
    plt.bar(class_names, counts)
    plt.title('Classification Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('classification_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Distribution plot saved as 'classification_distribution.png'")

def analyze_detection_distribution(detection_labels, data_config):
    """Analyze detection label distribution"""
    print("\n--- Detection Label Analysis ---")
    
    # Convert to numpy array
    detection_array = np.array(detection_labels)
    
    if len(detection_array) == 0:
        print("No detection labels found")
        return
    
    # Class distribution
    class_indices = detection_array[:, 0].astype(int)
    class_counts = Counter(class_indices)
    total_detections = len(class_indices)
    
    print(f"Total detection instances: {total_detections}")
    print("Detection class distribution:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = count / total_detections * 100
        class_name = data_config.get('names', [])[class_idx] if class_idx < len(data_config.get('names', [])) else f"Class_{class_idx}"
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Box size analysis
    box_sizes = detection_array[:, 3:5]  # width, height
    areas = box_sizes[:, 0] * box_sizes[:, 1]
    
    print(f"\nBox size statistics:")
    print(f"  Mean area: {np.mean(areas):.4f}")
    print(f"  Median area: {np.median(areas):.4f}")
    print(f"  Min area: {np.min(areas):.4f}")
    print(f"  Max area: {np.max(areas):.4f}")
    print(f"  Std area: {np.std(areas):.4f}")

def analyze_hyperparameters():
    """Analyze current hyperparameter configuration"""
    print("\n=== HYPERPARAMETER ANALYSIS ===")
    
    # Check current hyperparameter file
    hyp_files = ['yolov5c/data/hyps/hyp.custom.yaml', 'yolov5c/data/hyps/hyp.fixed.yaml']
    
    for hyp_file in hyp_files:
        if os.path.exists(hyp_file):
            print(f"\nAnalyzing: {hyp_file}")
            with open(hyp_file, 'r') as f:
                hyp_config = yaml.safe_load(f)
            
            print("Key parameters:")
            print(f"  cls_task: {hyp_config.get('cls_task', 'Not set')}")
            print(f"  classification_weight: {hyp_config.get('classification_weight', 'Not set')}")
            print(f"  lr0: {hyp_config.get('lr0', 'Not set')}")
            print(f"  warmup_epochs: {hyp_config.get('warmup_epochs', 'Not set')}")
            print(f"  box: {hyp_config.get('box', 'Not set')}")
            print(f"  cls: {hyp_config.get('cls', 'Not set')}")
            print(f"  fl_gamma: {hyp_config.get('fl_gamma', 'Not set')}")
            
            # Check augmentation settings
            aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'fliplr']
            print("Augmentation settings:")
            for param in aug_params:
                value = hyp_config.get(param, 'Not set')
                print(f"  {param}: {value}")

def main():
    """Main analysis function"""
    print("=== YOLOv5 Classification Performance Analysis ===\n")
    
    # Analyze dataset
    dataset_paths = [
        "Regurgitation-YOLODataset-Detection",
        "Regurgitation-YOLODataset-1new"
    ]
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            analyze_classification_labels(dataset_path)
            break
    else:
        print("WARNING: No dataset found to analyze")
    
    # Analyze hyperparameters
    analyze_hyperparameters()
    
    print("\n=== RECOMMENDATIONS ===")
    print("Based on the analysis, here are the key issues and recommendations:")
    print("\n1. CLASSIFICATION WEIGHT ISSUE:")
    print("   - Current cls_task weight (0.3) is too low")
    print("   - Recommendation: Increase to 0.5-0.7")
    print("\n2. DATA AUGMENTATION ISSUE:")
    print("   - Medical images need minimal augmentation")
    print("   - Recommendation: Disable most augmentations")
    print("\n3. WARMUP ISSUE:")
    print("   - Current warmup_epochs (5) may be insufficient")
    print("   - Recommendation: Increase to 10 epochs")
    print("\n4. CLASS IMBALANCE:")
    print("   - Check if classification labels are balanced")
    print("   - Consider using class weights if imbalanced")
    
    print("\nUse the fixed hyperparameter file: yolov5c/data/hyps/hyp.fixed.yaml")

if __name__ == "__main__":
    main()
