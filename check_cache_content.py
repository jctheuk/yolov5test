#!/usr/bin/env python3
"""
Check Cache Content
Examine the cache files to see if classification labels are stored correctly
"""

import os
import numpy as np
from pathlib import Path

def check_cache_files():
    """Check all cache files in the dataset"""
    
    dataset_path = Path("Regurgitation-YOLODataset-Detection")
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / "labels"
        if not labels_dir.exists():
            continue
            
        print(f"\n{'='*60}")
        print(f"CHECKING {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        # Look for cache files
        cache_files = list(labels_dir.glob("*.cache*"))
        if not cache_files:
            print(f"No cache files found in {labels_dir}")
            continue
        
        for cache_file in cache_files:
            print(f"\nCache file: {cache_file}")
            try:
                cache_data = np.load(cache_file, allow_pickle=True).item()
                print(f"Cache keys: {list(cache_data.keys())}")
                
                # Check if classification labels are in cache
                if 'classification_labels' in cache_data:
                    print("✓ Classification labels found in cache")
                else:
                    print("✗ Classification labels NOT found in cache")
                
                # Check a few samples
                sample_count = 0
                for key, value in cache_data.items():
                    if key not in ['hash', 'version', 'msgs', 'results'] and sample_count < 5:
                        if isinstance(value, list) and len(value) == 4:
                            labels, shapes, segments, classification = value
                            print(f"  Sample {key}:")
                            print(f"    Detection labels: {labels}")
                            print(f"    Classification: {classification}")
                        else:
                            print(f"  Sample {key}: {value}")
                        sample_count += 1
                
                # Check if all classification labels are the same
                if 'classification_labels' in cache_data:
                    all_classifications = []
                    for key, value in cache_data.items():
                        if key not in ['hash', 'version', 'msgs', 'results']:
                            if isinstance(value, list) and len(value) == 4:
                                _, _, _, classification = value
                                all_classifications.append(classification)
                    
                    if all_classifications:
                        unique_classifications = set(str(cls) for cls in all_classifications)
                        print(f"\nUnique classification labels in cache: {len(unique_classifications)}")
                        for cls in unique_classifications:
                            print(f"  {cls}")
                        
                        if len(unique_classifications) == 1:
                            print("⚠️  WARNING: All classification labels are the same!")
                        else:
                            print("✓ Multiple classification labels found")
                
            except Exception as e:
                print(f"Error loading cache: {e}")

def check_actual_labels():
    """Check actual label files to compare with cache"""
    
    print(f"\n{'='*60}")
    print("CHECKING ACTUAL LABEL FILES")
    print(f"{'='*60}")
    
    dataset_path = Path("Regurgitation-YOLODataset-Detection")
    train_labels_dir = dataset_path / "train" / "labels"
    
    if not train_labels_dir.exists():
        print("Train labels directory not found")
        return
    
    # Check first 10 label files
    files_checked = 0
    classifications = []
    
    for file in train_labels_dir.glob('*.txt'):
        if files_checked >= 10:
            break
            
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                cls_line = lines[1].strip()
                classifications.append(cls_line)
                print(f"  {file.name}: {cls_line}")
            
            files_checked += 1
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Check diversity
    unique_classifications = set(classifications)
    print(f"\nUnique classifications in actual files: {len(unique_classifications)}")
    for cls in unique_classifications:
        print(f"  {cls}")

def main():
    print("=== Cache Content Analysis ===")
    
    # Check cache files
    check_cache_files()
    
    # Check actual label files
    check_actual_labels()
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. If cache shows all same classification labels, delete cache files")
    print("2. Restart training to regenerate cache with correct labels")
    print("3. Check if the issue is resolved")

if __name__ == "__main__":
    main()
