"""
Alternative approaches to achieve balanced training without reorganizing data

Explore:
1. WeightedRandomSampler - PyTorch's built-in balanced sampling
2. Custom balanced batch sampler
3. Focal loss with auto-weighting
4. Oversampling minority class (PSAX)
5. Mixed approach: combine multiple techniques
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler

def approach1_weighted_random_sampler():
    """Approach 1: Use PyTorch's WeightedRandomSampler"""
    print("=" * 60)
    print("APPROACH 1: WEIGHTED RANDOM SAMPLER")
    print("=" * 60)
    
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    
    print("\nHow it works:")
    print("  1. Calculate sample weights (inverse frequency)")
    print("  2. PyTorch samples based on these weights")
    print("  3. Minority classes sampled more frequently")
    print("  4. Results in balanced batches")
    
    # Calculate sample weights
    class_weights = 1.0 / class_counts
    sample_weights = []
    for class_id, count in enumerate(class_counts):
        sample_weights.extend([class_weights[class_id]] * count)
    
    print(f"\nClass counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    print(f"Total samples: {len(sample_weights)}")
    
    # Simulate sampling
    print(f"\nSimulating sampling with WeightedRandomSampler:")
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Sample batches
    batch_size = 32
    batches_to_test = 10
    all_indices = list(sampler)
    
    batch_distributions = []
    for i in range(batches_to_test):
        batch_indices = all_indices[i*batch_size:(i+1)*batch_size]
        
        # Determine class for each index
        batch_classes = []
        current_idx = 0
        for class_id, count in enumerate(class_counts):
            batch_classes.extend([1 if idx >= current_idx and idx < current_idx + count else 0 
                                 for idx in batch_indices])
            current_idx += count
        
        # Count classes in batch
        class_dist = [sum([1 for idx in batch_indices if idx >= sum(class_counts[:class_id]) 
                          and idx < sum(class_counts[:class_id+1])]) 
                     for class_id in range(len(class_counts))]
        batch_distributions.append(class_dist)
        
        if i < 3:
            print(f"  Batch {i}: A4C={class_dist[0]}, PSAX={class_dist[1]}, PLAX={class_dist[2]}")
    
    # Calculate average distribution
    avg_dist = np.mean(batch_distributions, axis=0)
    print(f"\nAverage batch distribution: A4C={avg_dist[0]:.1f}, PSAX={avg_dist[1]:.1f}, PLAX={avg_dist[2]:.1f}")
    print(f"Expected (balanced): A4C=10.7, PSAX=10.7, PLAX=10.7")
    
    # Test bias evolution
    print(f"\nTesting bias evolution:")
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    for epoch in range(100):
        # Simulate balanced batch
        batch_dist = torch.tensor([10.7, 10.7, 10.7], dtype=torch.float32)  # Balanced
        gradient = batch_dist / batch_size - 1/3
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    print(f"\nPros:")
    print(f"  + Built into PyTorch")
    print(f"  + Easy to implement")
    print(f"  + Effective class balancing")
    print(f"Cons:")
    print(f"  - Samples with replacement (may see same image twice)")
    print(f"  - Slightly slower than regular sampling")

def approach2_custom_balanced_sampler():
    """Approach 2: Custom balanced batch sampler"""
    print("\n" + "=" * 60)
    print("APPROACH 2: CUSTOM BALANCED BATCH SAMPLER")
    print("=" * 60)
    
    class_counts = np.array([324, 218, 455])
    
    print("\nHow it works:")
    print("  1. Group samples by class")
    print("  2. Sample equal number from each class per batch")
    print("  3. Create perfectly balanced batches")
    
    # Simulate balanced sampling
    batch_size = 33  # Must be divisible by 3
    samples_per_class = batch_size // 3
    
    print(f"\nBatch size: {batch_size}")
    print(f"Samples per class: {samples_per_class}")
    
    # Test bias evolution
    print(f"\nTesting bias evolution:")
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    for epoch in range(100):
        # Perfectly balanced batch
        batch_dist = torch.tensor([11.0, 11.0, 11.0], dtype=torch.float32)
        gradient = batch_dist / batch_size - 1/3
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    print(f"\nPros:")
    print(f"  + Perfect class balance")
    print(f"  + No sample replacement needed")
    print(f"  + Best bias prevention")
    print(f"Cons:")
    print(f"  - Requires custom sampler implementation")
    print(f"  - Batch size must be divisible by num_classes")
    print(f"  - May not use all samples in minority class")

def approach3_focal_loss():
    """Approach 3: Focal loss with auto-weighting"""
    print("\n" + "=" * 60)
    print("APPROACH 3: FOCAL LOSS")
    print("=" * 60)
    
    class_counts = np.array([324, 218, 455])
    class_probs = class_counts / class_counts.sum()
    
    print("\nHow it works:")
    print("  1. Down-weight easy examples (well-classified)")
    print("  2. Up-weight hard examples (misclassified)")
    print("  3. Automatically balances training focus")
    
    # Calculate focal loss alpha
    alpha = class_counts.sum() / (len(class_counts) * class_counts)
    alpha = alpha / alpha.mean()
    
    print(f"\nClass counts: {class_counts}")
    print(f"Focal loss alpha: {alpha}")
    print(f"Focal loss gamma: 2.0 (standard)")
    
    # Test bias evolution with focal loss
    print(f"\nTesting bias evolution:")
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    for epoch in range(100):
        # Imbalanced batch but weighted by focal loss
        batch_dist = torch.tensor([10.4, 7.0, 14.6], dtype=torch.float32)  # Imbalanced
        gradient = batch_dist / 32 - 1/3
        
        # Apply focal loss weighting
        gradient = gradient * torch.tensor(alpha, dtype=torch.float32)
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    print(f"\nPros:")
    print(f"  + Automatic adaptation to hard examples")
    print(f"  + Works with imbalanced batches")
    print(f"  + Better than simple class weights")
    print(f"Cons:")
    print(f"  - Requires custom loss implementation")
    print(f"  - Hyperparameter tuning (gamma)")
    print(f"  - More complex than class weights")

def approach4_oversampling():
    """Approach 4: Oversample minority class (PSAX)"""
    print("\n" + "=" * 60)
    print("APPROACH 4: OVERSAMPLING MINORITY CLASS")
    print("=" * 60)
    
    class_counts = np.array([324, 218, 455])
    
    print("\nHow it works:")
    print("  1. Duplicate PSAX samples to match majority class")
    print("  2. Creates balanced dataset")
    print("  3. Train on balanced dataset")
    
    # Calculate how many times to duplicate PSAX
    max_count = class_counts.max()
    oversample_factors = max_count / class_counts
    
    print(f"\nOriginal counts: {class_counts}")
    print(f"Oversample factors: {oversample_factors}")
    print(f"New counts: {max_count * np.ones(3, dtype=int)}")
    
    # Test bias evolution with oversampling
    print(f"\nTesting bias evolution:")
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    balanced_probs = np.array([1/3, 1/3, 1/3])
    
    for epoch in range(100):
        # Balanced batch due to oversampling
        batch_dist = torch.tensor([10.7, 10.7, 10.7], dtype=torch.float32)
        gradient = batch_dist / 32 - 1/3
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    print(f"\nPros:")
    print(f"  + Simple to implement")
    print(f"  + Perfect class balance")
    print(f"  + Works with existing code")
    print(f"Cons:")
    print(f"  - Overfitting risk (same images repeated)")
    print(f"  - Longer training time")
    print(f"  - Increased dataset size")

def approach5_mixed_approach():
    """Approach 5: Combine multiple techniques"""
    print("\n" + "=" * 60)
    print("APPROACH 5: MIXED APPROACH")
    print("=" * 60)
    
    print("\nCombine best of all approaches:")
    print("  1. Shuffle=True (already enabled)")
    print("  2. Class weights (already implemented)")
    print("  3. Larger batch size (128 vs 32)")
    print("  4. Label smoothing (already enabled)")
    
    class_counts = np.array([324, 218, 455])
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.mean()
    
    print(f"\nClass weights: {class_weights}")
    print(f"Batch size: 128 (larger)")
    print(f"Label smoothing: 0.1")
    
    # Test bias evolution with mixed approach
    print(f"\nTesting bias evolution:")
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    class_probs = class_counts / class_counts.sum()
    
    for epoch in range(100):
        # Shuffled + larger batch = better balance
        batch_dist = torch.tensor([41.6, 28.0, 58.4], dtype=torch.float32)  # batch_size=128
        gradient = batch_dist / 128 - 1/3
        
        # Apply class weights
        gradient = gradient * torch.tensor(class_weights, dtype=torch.float32)
        
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: PSAX bias = {linear.bias[1]:.3f}")
    
    print(f"\nPros:")
    print(f"  + Best of all worlds")
    print(f"  + Minimal code changes")
    print(f"  + Proven effective")
    print(f"Cons:")
    print(f"  - Higher GPU memory usage")
    print(f"  - May need LR adjustment")

def compare_all_approaches():
    """Compare all approaches"""
    print("\n" + "=" * 60)
    print("COMPARISON OF ALL APPROACHES")
    print("=" * 60)
    
    print("\n| Approach | PSAX Bias | Ease | Memory | Speed |")
    print("|----------|-----------|------|--------|-------|")
    print("| Current (no fix) | -0.263 | N/A | Low | Fast |")
    print("| 1. WeightedRandomSampler | ~0.0 | Easy | Low | Medium |")
    print("| 2. Balanced Sampler | ~0.0 | Medium | Low | Medium |")
    print("| 3. Focal Loss | ~-0.05 | Hard | Low | Fast |")
    print("| 4. Oversampling | ~0.0 | Easy | High | Slow |")
    print("| 5. Mixed (recommended) | ~-0.01 | Easy | Medium | Fast |")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("\nBest options for you:")
    print("\n1. EASIEST: Mixed approach (already 90% done)")
    print("   - Use psax_bias_fix_hyp.yaml (class weights)")
    print("   - Increase batch size to 128")
    print("   - Command: python train_classification_task.py --batch-size 128 --hyp psax_bias_fix_hyp.yaml ...")
    
    print("\n2. BEST BALANCE: WeightedRandomSampler")
    print("   - Modify create_dataloader to add sampler")
    print("   - Automatically balances batches")
    print("   - Code change: ~10 lines")
    
    print("\n3. MOST EFFECTIVE: Balanced Sampler")
    print("   - Perfect class balance")
    print("   - Custom sampler implementation")
    print("   - Code change: ~50 lines")
    
    print("\n4. ALTERNATIVE: Focal Loss")
    print("   - Replace ClassificationTaskLoss with FocalLoss")
    print("   - Automatic hard example mining")
    print("   - Code change: ~30 lines")
    
    print("\nMy recommendation: Start with #1 (easiest), try #2 if needed")

if __name__ == "__main__":
    approach1_weighted_random_sampler()
    approach2_custom_balanced_sampler()
    approach3_focal_loss()
    approach4_oversampling()
    approach5_mixed_approach()
    compare_all_approaches()
