"""
Analyze why PSAX (class 1) has only 9% recall while A4C and PLAX have 44% and 52%

This is NOT normal - there's likely a code bug affecting class 1 specifically.
"""
import torch
import numpy as np
from pathlib import Path

def check_label_distribution():
    """Check if there's a label loading bug for PSAX"""
    print("=" * 60)
    print("ANALYZING PSAX (CLASS 1) BUG")
    print("=" * 60)
    
    # Read some label files to check format
    label_dir = Path("regurgitationV1/train/labels")
    label_files = list(label_dir.glob("*.txt"))[:100]
    
    class_counts = {0: 0, 1: 0, 2: 0}
    
    print("\nChecking label files for class distribution...")
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        # Find classification line
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:  # Classification line
                # One-hot: [1,0,0] or [0,1,0] or [0,0,1]
                try:
                    one_hot = [float(x) for x in parts]
                    class_idx = one_hot.index(1.0)
                    class_counts[class_idx] += 1
                except:
                    pass
    
    print(f"\nClass distribution in first 100 label files:")
    print(f"  A4C (0): {class_counts[0]} samples")
    print(f"  PSAX (1): {class_counts[1]} samples")
    print(f"  PLAX (2): {class_counts[2]} samples")
    
    # Check if there's an issue with PSAX labels
    if class_counts[1] < class_counts[0] * 0.5 or class_counts[1] < class_counts[2] * 0.5:
        print("\nWARNING: PSAX is significantly underrepresented in the dataset")
        print("  This could explain the poor PSAX performance")
    
    return class_counts

def check_one_hot_conversion_bug():
    """Check if there's a bug in one-hot to index conversion"""
    print("\n" + "=" * 60)
    print("CHECKING ONE-HOT CONVERSION LOGIC")
    print("=" * 60)
    
    # Test cases for one-hot conversion
    test_cases = [
        torch.tensor([[1., 0., 0.]]),  # A4C
        torch.tensor([[0., 1., 0.]]),  # PSAX
        torch.tensor([[0., 0., 1.]]),  # PLAX
    ]
    
    for i, one_hot in enumerate(test_cases):
        # Method 1: argmax (what the code uses)
        idx_argmax = one_hot.argmax(dim=-1)
        
        # Method 2: manual search
        idx_manual = torch.where(one_hot[0] == 1.0)[0][0]
        
        class_names = ['A4C', 'PSAX', 'PLAX']
        print(f"\nTest {i+1}: {one_hot.tolist()}")
        print(f"  argmax result: {idx_argmax.item()} ({class_names[idx_argmax.item()]})")
        print(f"  manual result: {idx_manual.item()} ({class_names[idx_manual.item()]})")
        
        if idx_argmax != idx_manual:
            print(f"  ERROR: Mismatch detected!")

def check_loss_computation_bias():
    """Check if CrossEntropyLoss is biased toward certain classes"""
    print("\n" + "=" * 60)
    print("CHECKING LOSS COMPUTATION FOR CLASS BIAS")
    print("=" * 60)
    
    # Simulate predictions for all classes
    batch_size = 10
    num_classes = 3
    
    # Test: What happens when model predicts class 1 (PSAX)?
    predictions = torch.randn(batch_size, num_classes)
    
    # Set predictions to favor class 1
    predictions[:, 1] += 2.0  # Boost class 1 logits
    
    # Test with different target scenarios
    print("\nScenario 1: All targets are class 1 (PSAX)")
    targets_psax = torch.ones(batch_size, dtype=torch.long)
    loss_psax = F.cross_entropy(predictions, targets_psax)
    print(f"  Loss when all targets are PSAX: {loss_psax.item():.4f}")
    
    print("\nScenario 2: All targets are class 0 (A4C)")
    targets_a4c = torch.zeros(batch_size, dtype=torch.long)
    loss_a4c = F.cross_entropy(predictions, targets_a4c)
    print(f"  Loss when all targets are A4C: {loss_a4c.item():.4f}")
    
    print("\nScenario 3: All targets are class 2 (PLAX)")
    targets_plax = torch.ones(batch_size, dtype=torch.long) * 2
    loss_plax = F.cross_entropy(predictions, targets_plax)
    print(f"  Loss when all targets are PLAX: {loss_plax.item():.4f}")
    
    print("\nLoss comparison:")
    print(f"  A4C loss: {loss_a4c.item():.4f}")
    print(f"  PSAX loss: {loss_psax.item():.4f}")
    print(f"  PLAX loss: {loss_plax.item():.4f}")
    
    if abs(loss_a4c - loss_psax) > 0.1 or abs(loss_a4c - loss_plax) > 0.1:
        print("\nWARNING: Significant loss differences detected!")
        print("  This suggests a potential bias in loss computation")

def check_argmax_bug_with_middle_class():
    """Check if there's a specific bug with middle class (index 1)"""
    print("\n" + "=" * 60)
    print("CHECKING FOR MIDDLE CLASS (INDEX 1) BUG")
    print("=" * 60)
    
    # Simulate what happens with PSAX labels
    print("\nTest: Converting PSAX labels from one-hot to indices")
    
    # Batch of PSAX labels (all class 1)
    psax_labels_onehot = torch.tensor([
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
    ])
    
    print(f"One-hot labels:\n{psax_labels_onehot}")
    
    # Convert using argmax (what the code does)
    indices = psax_labels_onehot.argmax(dim=-1)
    print(f"\nConverted indices: {indices}")
    print(f"Expected: tensor([1, 1, 1, 1])")
    
    if not torch.all(indices == 1):
        print("\nERROR: Argmax conversion failed for PSAX!")
    else:
        print("\nOK: Argmax conversion works correctly for PSAX")
    
    # Check if there's a floating point precision issue
    print("\nChecking floating point precision...")
    noisy_psax = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],  # Slightly noisy
        [0.0, 0.99, 0.01],  # Very slightly noisy
    ])
    
    indices_noisy = noisy_psax.argmax(dim=-1)
    print(f"Noisy one-hot:\n{noisy_psax}")
    print(f"Converted indices: {indices_noisy}")
    
    if not torch.all(indices_noisy == 1):
        print("\nERROR: Argmax fails with noisy labels!")
    else:
        print("\nOK: Argmax handles noisy labels correctly")

if __name__ == "__main__":
    check_label_distribution()
    check_one_hot_conversion_bug()
    check_argmax_bug_with_middle_class()
    check_loss_computation_bias()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nIf no bugs found here, the issue might be in:")
    print("  1. Model initialization bias toward classes 0 and 2")
    print("  2. Gradient flow bug affecting class 1 specifically")
    print("  3. Dataset issue - fewer PSAX samples in training data")
    print("  4. Validation set bias - different distribution than training")

