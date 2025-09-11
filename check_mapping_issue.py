#!/usr/bin/env python3
"""
Check for potential class mapping issues
"""

import os
import glob

def check_mapping_issue():
    """Check if there's a class mapping issue"""
    
    labels_dir = "Regurgitation-YOLODataset-Detection/valid/labels"
    
    print("=" * 80)
    print("CHECKING FOR CLASS MAPPING ISSUES")
    print("=" * 80)
    
    # Current mapping from data.yaml
    print("Current mapping in data.yaml:")
    print("  cls_names: ['A4C', 'PSAX', 'PLAX']")
    print("  This means: 0=A4C, 1=PSAX, 2=PLAX")
    print()
    
    # Check some specific files
    test_files = [
        "a2lrwqduZsKc-unnamed_1_1.mp4-31.txt",  # PSAX
        "aGdjwqtqa8Kb-unnamed_1_1.mp4-62.txt",  # A4C
        "a2ZnwqdsaMKZ-unnamed_1_1.mp4-2.txt"    # PLAX
    ]
    
    print("Checking specific files:")
    for filename in test_files:
        filepath = os.path.join(labels_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            lines = content.split('\n')
            if len(lines) >= 2:
                detection_line = lines[0]
                classification_line = lines[1]
                
                print(f"\n{filename}:")
                print(f"  Detection: {detection_line}")
                print(f"  Classification: {classification_line}")
                
                # Parse classification
                parts = classification_line.split()
                if len(parts) == 3:
                    a4c, psax, plax = map(int, parts)
                    if a4c == 1:
                        print(f"  → Labeled as: A4C (index 0)")
                    elif psax == 1:
                        print(f"  → Labeled as: PSAX (index 1)")
                    elif plax == 1:
                        print(f"  → Labeled as: PLAX (index 2)")
                    else:
                        print(f"  → ERROR: No valid classification!")
        else:
            print(f"\n{filename}: File not found")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Count each class
    a4c_count = 0
    psax_count = 0
    plax_count = 0
    total = 0
    
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                classification_line = lines[1].strip()
                parts = classification_line.split()
                
                if len(parts) == 3:
                    a4c, psax, plax = map(int, parts)
                    total += 1
                    
                    if a4c == 1:
                        a4c_count += 1
                    elif psax == 1:
                        psax_count += 1
                    elif plax == 1:
                        plax_count += 1
                        
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    print(f"Validation set distribution:")
    print(f"  A4C (index 0): {a4c_count} samples ({a4c_count/total*100:.1f}%)")
    print(f"  PSAX (index 1): {psax_count} samples ({psax_count/total*100:.1f}%)")
    print(f"  PLAX (index 2): {plax_count} samples ({plax_count/total*100:.1f}%)")
    print(f"  Total: {total} samples")
    
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES")
    print("=" * 80)
    
    # Check if the mapping might be wrong
    if a4c_count < 30:  # A4C is very rare
        print("⚠️  A4C samples are very rare - this could indicate:")
        print("   1. A4C is genuinely rare in the dataset")
        print("   2. A4C might be mislabeled as another class")
        print("   3. The class mapping might be wrong")
    
    if psax_count > 70:  # PSAX is very common
        print("⚠️  PSAX samples are very common - this could indicate:")
        print("   1. PSAX is genuinely common in the dataset")
        print("   2. Some A4C/PLAX might be mislabeled as PSAX")
        print("   3. The class mapping might be wrong")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. Manually check a few images from each class to verify mapping")
    print("2. If mapping is wrong, update data.yaml cls_names")
    print("3. If mapping is correct, the issue is class imbalance")
    print("4. Consider collecting more A4C samples")

if __name__ == "__main__":
    check_mapping_issue()
