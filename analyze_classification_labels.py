#!/usr/bin/env python3
"""
Analyze Classification Labels in YOLO Dataset
Check if all classification labels are [0,1,0] or if there's a data loading issue
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_classification_labels(dataset_path):
    """Analyze classification labels in the dataset"""
    
    dataset_path = Path(dataset_path)
    splits = ['train', 'valid', 'test']
    
    results = {}
    
    for split in splits:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist")
            continue
            
        print(f"\n=== Analyzing {split} split ===")
        
        class_counts = {'PSAX': 0, 'PLAX': 0, 'A4C': 0, 'ERROR': 0, 'EMPTY': 0}
        label_samples = {'PSAX': [], 'PLAX': [], 'A4C': [], 'ERROR': []}
        
        total_files = 0
        files_with_classification = 0
        
        for file in labels_dir.glob('*.txt'):
            total_files += 1
            
            try:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    
                # Check if file has at least 2 lines
                if len(lines) < 2:
                    class_counts['EMPTY'] += 1
                    continue
                
                # Get classification line (second line)
                cls_line = lines[1].strip()
                
                # Parse classification label
                if cls_line == '1 0 0':
                    class_counts['PSAX'] += 1
                    label_samples['PSAX'].append(str(file))
                elif cls_line == '0 1 0':
                    class_counts['PLAX'] += 1
                    label_samples['PLAX'].append(str(file))
                elif cls_line == '0 0 1':
                    class_counts['A4C'] += 1
                    label_samples['A4C'].append(str(file))
                else:
                    class_counts['ERROR'] += 1
                    label_samples['ERROR'].append(f"{file}: {cls_line}")
                
                files_with_classification += 1
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                class_counts['ERROR'] += 1
        
        results[split] = {
            'class_counts': class_counts,
            'label_samples': label_samples,
            'total_files': total_files,
            'files_with_classification': files_with_classification
        }
        
        # Print results
        print(f"Total files: {total_files}")
        print(f"Files with classification labels: {files_with_classification}")
        print(f"Classification distribution:")
        for cls, count in class_counts.items():
            if count > 0:
                print(f"  {cls}: {count}")
        
        # Show some samples
        for cls, samples in label_samples.items():
            if samples:
                print(f"\n{cls} samples (first 5):")
                for sample in samples[:5]:
                    print(f"  {sample}")
                if len(samples) > 5:
                    print(f"  ... and {len(samples) - 5} more")
    
    return results

def check_dataloader_reading(dataset_path):
    """Check how the dataloader reads the classification labels"""
    
    print("\n=== Checking Dataloader Reading ===")
    
    # Simulate the dataloader reading process
    labels_dir = Path(dataset_path) / 'train' / 'labels'
    
    if not labels_dir.exists():
        print("Train labels directory not found")
        return
    
    # Check first 10 files
    files_checked = 0
    for file in labels_dir.glob('*.txt'):
        if files_checked >= 10:
            break
            
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            print(f"\nFile: {file.name}")
            print(f"Number of lines: {len(lines)}")
            
            if len(lines) >= 1:
                print(f"Line 1 (detection): {lines[0].strip()}")
            if len(lines) >= 2:
                print(f"Line 2 (classification): {lines[1].strip()}")
                
                # Try to parse as list
                try:
                    cls_values = [int(x) for x in lines[1].strip().split()]
                    print(f"Parsed as list: {cls_values}")
                except:
                    print(f"Could not parse as list")
            
            files_checked += 1
            
        except Exception as e:
            print(f"Error reading {file}: {e}")

def create_visualization(results):
    """Create visualization of classification distribution"""
    
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (split, data) in enumerate(results.items()):
        class_counts = data['class_counts']
        
        # Remove ERROR and EMPTY for visualization
        viz_counts = {k: v for k, v in class_counts.items() if k not in ['ERROR', 'EMPTY']}
        
        if viz_counts:
            classes = list(viz_counts.keys())
            counts = list(viz_counts.values())
            
            axes[i].bar(classes, counts)
            axes[i].set_title(f'{split.capitalize()} Split')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for j, count in enumerate(counts):
                axes[i].text(j, count + max(counts) * 0.01, str(count), 
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('classification_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    dataset_path = "Regurgitation-YOLODataset-Detection"
    
    print("=== Classification Label Analysis ===")
    print(f"Dataset path: {dataset_path}")
    
    # Analyze classification labels
    results = analyze_classification_labels(dataset_path)
    
    # Check dataloader reading
    check_dataloader_reading(dataset_path)
    
    # Create visualization
    try:
        create_visualization(results)
        print("\nVisualization saved as 'classification_distribution.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Summary
    print("\n=== SUMMARY ===")
    for split, data in results.items():
        print(f"\n{split.upper()} SPLIT:")
        class_counts = data['class_counts']
        total_valid = sum([v for k, v in class_counts.items() if k not in ['ERROR', 'EMPTY']])
        
        if total_valid > 0:
            print(f"Total valid classification labels: {total_valid}")
            for cls, count in class_counts.items():
                if cls not in ['ERROR', 'EMPTY'] and count > 0:
                    percentage = (count / total_valid) * 100
                    print(f"  {cls}: {count} ({percentage:.1f}%)")
        else:
            print("No valid classification labels found!")
        
        if class_counts['ERROR'] > 0:
            print(f"  ERRORS: {class_counts['ERROR']}")
        if class_counts['EMPTY'] > 0:
            print(f"  EMPTY FILES: {class_counts['EMPTY']}")

if __name__ == "__main__":
    main()
