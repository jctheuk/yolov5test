"""
Analyze why shuffle=True is already enabled but PSAX bias is still -0.263

The issue might be:
1. Shuffle works at batch level, but class distribution in dataset is still imbalanced
2. Your dataset structure vs ImageFolder structure
3. Batch size effect (32 vs 128)
4. Need to combine shuffle + class weights for best results
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_shuffle_limitations():
    """Analyze why shuffle alone isn't enough"""
    print("=" * 60)
    print("ANALYZING SHUFFLE LIMITATIONS")
    print("=" * 60)
    
    print("\n1. SHUFFLE IS ALREADY ENABLED:")
    print("   train_classification_task.py line 834: shuffle=True")
    print("   But PSAX bias is still -0.263")
    
    print("\n2. WHY SHUFFLE ALONE ISN'T ENOUGH:")
    print("   - Shuffle randomizes order within batches")
    print("   - But overall class distribution is still [32.5%, 21.9%, 45.6%]")
    print("   - Each batch still reflects this imbalanced distribution")
    print("   - Bias still evolves toward majority class (PLAX)")
    
    print("\n3. IMAGEFOLDER vs LOADIMAGESANDLABELS:")
    print("   ImageFolder (classify/):")
    print("     - Each class in separate folder")
    print("     - Shuffle samples across ALL classes")
    print("     - Better batch balancing")
    print("   LoadImagesAndLabels (yolov5c):")
    print("     - Sequential file loading")
    print("     - Shuffle within dataset order")
    print("     - Less effective balancing")

def test_batch_size_effect():
    """Test how batch size affects shuffle effectiveness"""
    print("\n" + "=" * 60)
    print("TESTING BATCH SIZE EFFECT ON SHUFFLE")
    print("=" * 60)
    
    # Your class distribution
    class_counts = np.array([324, 218, 455])
    class_probs = class_counts / class_counts.sum()
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size {batch_size}:")
        
        # Simulate multiple batches with shuffle
        torch.manual_seed(42)
        linear = nn.Linear(1280, 3)
        nn.init.constant_(linear.bias, 0.0)
        
        lr = 0.01
        for epoch in range(50):  # Fewer epochs for comparison
            # Shuffled batch - each batch has random distribution
            class_counts_batch = np.random.multinomial(batch_size, class_probs)
            batch_dist = torch.tensor(class_counts_batch, dtype=torch.float32)
            
            # Simulate gradient
            gradient = batch_dist / batch_size - 1/3
            with torch.no_grad():
                linear.bias += lr * gradient
        
        final_bias = linear.bias.data
        print(f"  Final PSAX bias: {final_bias[1]:.3f}")
        
        if batch_size == 32:
            print(f"    (Your current batch size)")
        elif batch_size == 128:
            print(f"    (Original classify/ batch size)")

def test_combined_approach():
    """Test shuffle + class weights combination"""
    print("\n" + "=" * 60)
    print("TESTING SHUFFLE + CLASS WEIGHTS COMBINATION")
    print("=" * 60)
    
    class_counts = np.array([324, 218, 455])
    class_probs = class_counts / class_counts.sum()
    
    # Calculate class weights
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.mean()
    
    print(f"Class weights: {class_weights}")
    
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    batch_size = 32
    
    for epoch in range(100):
        # Shuffled batch
        class_counts_batch = np.random.multinomial(batch_size, class_probs)
        batch_dist = torch.tensor(class_counts_batch, dtype=torch.float32)
        
        # Gradient with class weights
        gradient = batch_dist / batch_size - 1/3
        gradient = gradient * torch.tensor(class_weights, dtype=torch.float32)
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias (shuffle + weights): [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
    print(f"PSAX bias: {final_bias[1]:.3f} (should be close to 0)")

def recommendations():
    """Provide recommendations"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n1. CURRENT STATUS:")
    print("   - Shuffle is already enabled (shuffle=True)")
    print("   - But PSAX bias is still -0.263")
    print("   - Need additional measures")
    
    print("\n2. SOLUTIONS (in order of effectiveness):")
    print("   A. Use class weights (already implemented)")
    print("      - psax_bias_fix_hyp.yaml")
    print("      - Most effective solution")
    print("   B. Increase batch size")
    print("      - Change from 32 to 128")
    print("      - Better batch balancing")
    print("   C. Combine shuffle + class weights")
    print("      - Best of both worlds")
    
    print("\n3. IMPLEMENTATION:")
    print("   Option 1: Test class weights (easiest)")
    print("     python train_classification_task.py --hyp psax_bias_fix_hyp.yaml ...")
    print("   Option 2: Test larger batch size")
    print("     python train_classification_task.py --batch-size 128 ...")
    print("   Option 3: Test both combined")
    print("     python train_classification_task.py --batch-size 128 --hyp psax_bias_fix_hyp.yaml ...")

if __name__ == "__main__":
    analyze_shuffle_limitations()
    test_batch_size_effect()
    test_combined_approach()
    recommendations()
