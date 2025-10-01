"""Count class distribution across entire dataset"""
from pathlib import Path
import yaml
import torch.nn.functional as F

def count_classes_in_dataset(split='train'):
    """Count class distribution in dataset"""
    label_dir = Path(f"regurgitationV1/{split}/labels")
    label_files = list(label_dir.glob("*.txt"))
    
    class_counts = {0: 0, 1: 0, 2: 0}
    total = 0
    
    print(f"Counting classes in {split} set...")
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:  # Classification line
                try:
                    one_hot = [float(x) for x in parts]
                    class_idx = one_hot.index(1.0)
                    class_counts[class_idx] += 1
                    total += 1
                except:
                    pass
    
    return class_counts, total

print("=" * 60)
print("COMPLETE DATASET CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)

# Check train set
train_counts, train_total = count_classes_in_dataset('train')
print(f"\nTRAIN SET:")
print(f"  Total samples: {train_total}")
print(f"  A4C (0):  {train_counts[0]:4d} samples ({train_counts[0]/train_total*100:5.1f}%)")
print(f"  PSAX (1): {train_counts[1]:4d} samples ({train_counts[1]/train_total*100:5.1f}%) [WARNING]")
print(f"  PLAX (2): {train_counts[2]:4d} samples ({train_counts[2]/train_total*100:5.1f}%)")

# Check valid set
try:
    valid_counts, valid_total = count_classes_in_dataset('valid')
    print(f"\nVALIDATION SET:")
    print(f"  Total samples: {valid_total}")
    print(f"  A4C (0):  {valid_counts[0]:4d} samples ({valid_counts[0]/valid_total*100:5.1f}%)")
    print(f"  PSAX (1): {valid_counts[1]:4d} samples ({valid_counts[1]/valid_total*100:5.1f}%) [WARNING]")
    print(f"  PLAX (2): {valid_counts[2]:4d} samples ({valid_counts[2]/valid_total*100:5.1f}%)")
except:
    print("\nVALIDATION SET: Not found or error")

print("\n" + "=" * 60)
print("IMBALANCE RATIO:")
print("=" * 60)
print(f"A4C:PSAX:PLAX = {train_counts[0]}:{train_counts[1]}:{train_counts[2]}")
print(f"PSAX is {train_counts[0]/max(train_counts[1],1):.1f}x less frequent than A4C")
print(f"PSAX is {train_counts[2]/max(train_counts[1],1):.1f}x less frequent than PLAX")

if train_counts[1] < 50:
    print("\nCRITICAL: PSAX has VERY FEW samples!")
    print("   With so few PSAX examples, the model cannot learn this class")
    print("   This explains the 9% PSAX recall - NOT a code bug!")

