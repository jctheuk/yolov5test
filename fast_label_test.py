#!/usr/bin/env python3
"""
Fast Label Format Test
Quick check of label files to identify dataset vs dataloader issues
"""

import yaml
from pathlib import Path

def fast_test():
    """Quick test focusing on key issues"""
    print("FAST LABEL FORMAT TEST")
    print("=" * 40)
    
    # Test 1: Check data.yaml
    print("TEST 1: Data Configuration")
    print("-" * 25)
    
    data_yaml = Path("regurgitationV1/data.yaml")
    if not data_yaml.exists():
        print("ERROR: data.yaml not found")
        return False
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("OK: Data configuration:")
    print(f"   Detection classes: {data_config.get('nc', 'NOT FOUND')}")
    print(f"   Detection names: {data_config.get('names', 'NOT FOUND')}")
    print(f"   Classification classes: {data_config.get('num_cls', 'NOT FOUND')}")
    print(f"   Classification names: {data_config.get('cls_names', 'NOT FOUND')}")
    
    if 'num_cls' not in data_config or 'cls_names' not in data_config:
        print("ERROR: Missing classification configuration")
        return False
    
    # Test 2: Check just 3 label files
    print("\nTEST 2: Label File Format (3 files)")
    print("-" * 25)
    
    label_dir = Path("regurgitationV1/train/labels")
    if not label_dir.exists():
        print("ERROR: Label directory not found")
        return False
    
    # Check first 3 label files only
    label_files = list(label_dir.glob("*.txt"))[:3]
    print(f"Checking {len(label_files)} label files...")
    
    classification_labels_found = 0
    
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
                    break
            
            if classification_line:
                classification_labels_found += 1
                print(f"   OK {label_file.name}: Classification={classification_line}")
            else:
                print(f"   ERROR {label_file.name}: No classification label found")
                
        except Exception as e:
            print(f"   ERROR reading {label_file.name}: {e}")
    
    print(f"\nSummary:")
    print(f"   Files with classification labels: {classification_labels_found}/{len(label_files)}")
    
    if classification_labels_found == 0:
        print("ERROR: NO CLASSIFICATION LABELS FOUND!")
        print("   This is the root cause of poor classification performance.")
        return False
    
    # Test 3: Check one specific file in detail
    print("\nTEST 3: Detailed File Analysis")
    print("-" * 25)
    
    sample_file = Path("regurgitationV1/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            lines = f.read().strip().splitlines()
        
        print(f"Sample file: {sample_file.name}")
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
                    class_names = ['A4C', 'PSAX', 'PLAX']
                    print(f"              -> Class {class_idx}: {class_names[class_idx]}")
                else:
                    print(f"   Line {i}: Unknown format - {parts}")
            else:
                print(f"   Line {i}: Empty line")
    
    print("\nFast test complete!")
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if classification_labels_found > 0:
        print("OK: Classification labels are present in label files")
        print("OK: Label format appears correct (3-line format)")
        print("\nCONCLUSION: Dataset format is CORRECT!")
        print("   The issue is likely in the DATALOADER or MODEL ARCHITECTURE.")
        print("   Classification labels are properly formatted in the files.")
        print("\nNEXT STEPS:")
        print("   1. Check if dataloader is loading classification labels correctly")
        print("   2. Verify model is receiving classification labels")
        print("   3. Check if loss function is using classification labels")
        
    else:
        print("ERROR: No classification labels found in any files!")
        print("   This is the root cause of the poor classification performance.")
        print("   The model is training on default labels (all A4C).")
    
    return classification_labels_found > 0

if __name__ == "__main__":
    fast_test()

