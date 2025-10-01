#!/usr/bin/env python3
"""
Simple Label Format Test
Directly check label files without dataloader complications
"""

import yaml
from pathlib import Path
from collections import Counter

def test_label_files_directly():
    """Test label files directly without dataloader"""
    print("🚀 SIMPLE LABEL FORMAT TEST")
    print("=" * 50)
    
    # Test 1: Check data.yaml
    print("🔍 TEST 1: Data Configuration")
    print("-" * 30)
    
    data_yaml = Path("regurgitationV1/data.yaml")
    if not data_yaml.exists():
        print("❌ data.yaml not found")
        return False
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("✅ Data configuration:")
    print(f"   Detection classes: {data_config.get('nc', 'NOT FOUND')}")
    print(f"   Detection names: {data_config.get('names', 'NOT FOUND')}")
    print(f"   Classification classes: {data_config.get('num_cls', 'NOT FOUND')}")
    print(f"   Classification names: {data_config.get('cls_names', 'NOT FOUND')}")
    
    if 'num_cls' not in data_config or 'cls_names' not in data_config:
        print("❌ Missing classification configuration")
        return False
    
    # Test 2: Check label file format
    print("\n🔍 TEST 2: Label File Format")
    print("-" * 30)
    
    label_dir = Path("regurgitationV1/train/labels")
    if not label_dir.exists():
        print("❌ Label directory not found")
        return False
    
    # Check first 10 label files
    label_files = list(label_dir.glob("*.txt"))[:10]
    print(f"📁 Checking {len(label_files)} label files...")
    
    classification_labels_found = 0
    detection_labels_found = 0
    malformed_files = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
            
            detection_lines = []
            classification_line = None
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 5:  # Detection line
                    detection_lines.append(parts)
                elif len(parts) == 3:  # Classification line
                    classification_line = parts
                else:
                    malformed_files += 1
                    break
            
            if detection_lines:
                detection_labels_found += 1
            if classification_line:
                classification_labels_found += 1
                print(f"   ✅ {label_file.name}: Detection={len(detection_lines)}, Classification={classification_line}")
            else:
                print(f"   ❌ {label_file.name}: No classification label found")
                
        except Exception as e:
            print(f"   ❌ Error reading {label_file.name}: {e}")
            malformed_files += 1
    
    print(f"\n📊 Summary:")
    print(f"   Files with detection labels: {detection_labels_found}/{len(label_files)}")
    print(f"   Files with classification labels: {classification_labels_found}/{len(label_files)}")
    print(f"   Malformed files: {malformed_files}")
    
    if classification_labels_found == 0:
        print("❌ NO CLASSIFICATION LABELS FOUND IN ANY FILES!")
        return False
    
    # Test 3: Check class distribution
    print("\n🔍 TEST 3: Class Distribution")
    print("-" * 30)
    
    class_counts = Counter()
    total_files = 0
    files_without_classification = 0
    
    # Check all label files for class distribution
    for label_file in label_dir.glob("*.txt"):
        total_files += 1
        try:
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
            
            classification_line = None
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 3:  # Classification line
                    classification_line = parts
                    break
            
            if classification_line:
                # Convert one-hot to class index
                try:
                    class_idx = classification_line.index('1')
                    class_counts[class_idx] += 1
                except ValueError:
                    files_without_classification += 1
            else:
                files_without_classification += 1
                
        except Exception as e:
            files_without_classification += 1
    
    class_names = ['A4C', 'PSAX', 'PLAX']
    print(f"📈 Class Distribution (out of {total_files} files):")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"   {name} (class {i}): {count} files ({percentage:.1f}%)")
    
    print(f"   Files without classification: {files_without_classification}")
    
    # Check for imbalance
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 2:
            print(f"⚠️  Class imbalance detected: ratio {imbalance_ratio:.1f}:1")
        else:
            print("✅ Class distribution is balanced")
    
    # Test 4: Check specific label file format
    print("\n🔍 TEST 4: Detailed Label File Analysis")
    print("-" * 30)
    
    sample_file = Path("regurgitationV1/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            lines = f.read().strip().splitlines()
        
        print(f"📁 Sample file: {sample_file.name}")
        print(f"   Total lines: {len(lines)}")
        
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.split()
                if len(parts) == 5:
                    print(f"   Line {i}: Detection - class {parts[0]}, bbox {parts[1:]}")
                elif len(parts) == 3:
                    print(f"   Line {i}: Classification - {parts} (one-hot)")
                    # Show which class this represents
                    class_idx = parts.index('1')
                    print(f"              → Class {class_idx}: {class_names[class_idx]}")
                else:
                    print(f"   Line {i}: Unknown format - {parts}")
            else:
                print(f"   Line {i}: Empty line")
    
    print("\n✅ Label format analysis complete!")
    
    # Summary
    print("\n📊 FINAL SUMMARY:")
    print("=" * 50)
    
    if classification_labels_found > 0:
        print("✅ Classification labels are present in label files")
        print("✅ Label format appears correct (3-line format)")
        print("✅ Class distribution analysis completed")
        
        if files_without_classification == 0:
            print("✅ All files have classification labels")
        else:
            print(f"⚠️  {files_without_classification} files missing classification labels")
        
        print("\n🎯 CONCLUSION: Dataset format appears correct!")
        print("   The issue is likely in the dataloader or model architecture.")
        print("   Classification labels are properly formatted in the files.")
        
    else:
        print("❌ No classification labels found in any files!")
        print("   This is the root cause of the poor classification performance.")
        print("   The model is training on default labels (all A4C).")
    
    return classification_labels_found > 0

if __name__ == "__main__":
    test_label_files_directly()

