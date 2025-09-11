#!/usr/bin/env python3
"""
Compare training and validation set distributions
"""

import os
import glob

def analyze_dataset_split(dataset_path, split_name):
    labels_dir = os.path.join(dataset_path, "labels")
    
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
    
    return a4c_count, psax_count, plax_count, total

def main():
    dataset_base = "Regurgitation-YOLODataset-Detection"
    
    print("=" * 80)
    print("DATASET CLASSIFICATION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Analyze training set
    train_a4c, train_psax, train_plax, train_total = analyze_dataset_split(
        os.path.join(dataset_base, "train"), "Training"
    )
    
    # Analyze validation set
    val_a4c, val_psax, val_plax, val_total = analyze_dataset_split(
        os.path.join(dataset_base, "valid"), "Validation"
    )
    
    # Print results
    print(f"\nTRAINING SET:")
    print(f"  Total files: {train_total}")
    print(f"  A4C: {train_a4c} ({train_a4c/train_total*100:.1f}%)")
    print(f"  PSAX: {train_psax} ({train_psax/train_total*100:.1f}%)")
    print(f"  PLAX: {train_plax} ({train_plax/train_total*100:.1f}%)")
    
    print(f"\nVALIDATION SET:")
    print(f"  Total files: {val_total}")
    print(f"  A4C: {val_a4c} ({val_a4c/val_total*100:.1f}%)")
    print(f"  PSAX: {val_psax} ({val_psax/val_total*100:.1f}%)")
    print(f"  PLAX: {val_plax} ({val_plax/val_total*100:.1f}%)")
    
    # Calculate ratios
    print(f"\nCLASS DISTRIBUTION RATIOS:")
    print(f"  A4C: Training={train_a4c/train_total*100:.1f}% vs Validation={val_a4c/val_total*100:.1f}%")
    print(f"  PSAX: Training={train_psax/train_total*100:.1f}% vs Validation={val_psax/val_total*100:.1f}%")
    print(f"  PLAX: Training={train_plax/train_total*100:.1f}% vs Validation={val_plax/val_total*100:.1f}%")
    
    # Check for imbalance
    print(f"\nCLASS IMBALANCE ANALYSIS:")
    train_ratios = [train_a4c/train_total, train_psax/train_total, train_plax/train_total]
    val_ratios = [val_a4c/val_total, val_psax/val_total, val_plax/val_total]
    
    train_imbalance = max(train_ratios) / min(train_ratios)
    val_imbalance = max(val_ratios) / min(val_ratios)
    
    print(f"  Training set imbalance ratio: {train_imbalance:.2f}:1")
    print(f"  Validation set imbalance ratio: {val_imbalance:.2f}:1")
    
    if train_imbalance > 3:
        print("  ⚠️  Training set has significant class imbalance!")
    if val_imbalance > 3:
        print("  ⚠️  Validation set has significant class imbalance!")
    
    # A4C specific analysis
    a4c_ratio_diff = abs(train_a4c/train_total - val_a4c/val_total)
    print(f"\nA4C SPECIFIC ANALYSIS:")
    print(f"  A4C ratio difference between train/val: {a4c_ratio_diff*100:.1f}%")
    
    if a4c_ratio_diff > 0.05:  # 5% difference
        print("  ⚠️  Significant difference in A4C distribution between train/val!")
        print("  This could explain poor A4C performance in validation.")
    
    if val_a4c/val_total < 0.2:  # Less than 20%
        print("  ⚠️  A4C is underrepresented in validation set (< 20%)")
        print("  This makes it harder for the model to learn A4C features.")

if __name__ == "__main__":
    main()
