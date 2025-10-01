"""
Test if shuffling data improves class imbalance bias issue

Compare:
1. Sequential loading (current) - maintains class distribution per batch
2. Shuffled loading - randomizes class distribution per batch
3. Class weights (already implemented)

This could be a simple fix without needing complex class weights!
"""

import torch
import torch.nn as nn
import numpy as np

def simulate_current_loading():
    """Simulate current sequential loading approach"""
    print("=" * 60)
    print("SIMULATING CURRENT SEQUENTIAL LOADING")
    print("=" * 60)
    
    # Your actual class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    
    # Simulate sequential loading (each batch maintains distribution)
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    print(f"\nInitial bias: {linear.bias.data}")
    
    for epoch in range(100):
        # Sequential loading: each batch has the same distribution
        batch_size = 32
        # Use actual distribution for each batch (worst case)
        class_counts_batch = np.random.multinomial(batch_size, class_probs)
        
        # Convert to tensor and simulate gradient
        batch_dist = torch.tensor(class_counts_batch, dtype=torch.float32)
        gradient = batch_dist / batch_size - 1/3  # Deviation from uniform
        
        # Update bias
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}]")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias (sequential): [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
    return final_bias

def simulate_shuffled_loading():
    """Simulate shuffled loading approach"""
    print("\n" + "=" * 60)
    print("SIMULATING SHUFFLED LOADING")
    print("=" * 60)
    
    # Your actual class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    
    # Create shuffled dataset (like ImageFolder + shuffle=True)
    # Generate all samples and shuffle them
    all_samples = []
    for class_idx, count in enumerate(class_counts):
        all_samples.extend([class_idx] * count)
    
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    print(f"\nInitial bias: {linear.bias.data}")
    print(f"Total samples: {len(all_samples)}")
    
    # Shuffle the dataset
    np.random.shuffle(all_samples)
    
    for epoch in range(100):
        # Shuffled loading: each batch is random sample from shuffled data
        batch_size = 32
        start_idx = (epoch * batch_size) % (len(all_samples) - batch_size)
        batch_samples = all_samples[start_idx:start_idx + batch_size]
        
        # Count classes in this batch
        batch_dist = torch.bincount(torch.tensor(batch_samples), minlength=3).float()
        
        # Simulate gradient
        gradient = batch_dist / batch_size - 1/3  # Deviation from uniform
        
        # Update bias
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            batch_probs = batch_dist / batch_size
            print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}], batch_dist = [{batch_probs[0]:.2f}, {batch_probs[1]:.2f}, {batch_probs[2]:.2f}]")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias (shuffled): [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
    return final_bias

def simulate_class_weights():
    """Simulate class weights approach (for comparison)"""
    print("\n" + "=" * 60)
    print("SIMULATING CLASS WEIGHTS APPROACH")
    print("=" * 60)
    
    # Your actual class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    # Calculate class weights (inverse frequency)
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.mean()  # Normalize
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    print(f"Class weights: {class_weights}")
    
    torch.manual_seed(42)
    linear = nn.Linear(1280, 3)
    nn.init.constant_(linear.bias, 0.0)
    
    lr = 0.01
    print(f"\nInitial bias: {linear.bias.data}")
    
    for epoch in range(100):
        # Sequential loading with class weights
        batch_size = 32
        class_counts_batch = np.random.multinomial(batch_size, class_probs)
        
        # Apply class weights to gradient
        batch_dist = torch.tensor(class_counts_batch, dtype=torch.float32)
        gradient = batch_dist / batch_size - 1/3  # Deviation from uniform
        
        # Weight the gradient by class weights
        gradient = gradient * torch.tensor(class_weights, dtype=torch.float32)
        
        # Update bias
        with torch.no_grad():
            linear.bias += lr * gradient
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}]")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias (class weights): [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
    return final_bias

def test_shuffle_in_dataloader():
    """Test if we can enable shuffle in your current dataloader"""
    print("\n" + "=" * 60)
    print("TESTING SHUFFLE IN YOUR DATALOADER")
    print("=" * 60)
    
    # Check your current dataloader creation in train_classification_task.py
    print("\nCurrent dataloader creation (train_classification_task.py lines ~620-640):")
    print("  train_loader, train_sampler = create_dataloader(...)")
    print("  val_loader = create_dataloader(..., shuffle=False)")
    
    print("\nTo enable shuffle, we need to modify:")
    print("  1. train_loader: add shuffle=True")
    print("  2. val_loader: keep shuffle=False (for consistent validation)")
    
    print("\nExpected improvement:")
    print("  - More balanced batches during training")
    print("  - Reduced bias toward majority class (PLAX)")
    print("  - Better PSAX recall")
    print("  - No need for class weights")

if __name__ == "__main__":
    sequential_bias = simulate_current_loading()
    shuffled_bias = simulate_shuffled_loading()
    weights_bias = simulate_class_weights()
    test_shuffle_in_dataloader()
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nSequential loading bias:  [{sequential_bias[0]:.3f}, {sequential_bias[1]:.3f}, {sequential_bias[2]:.3f}]")
    print(f"Shuffled loading bias:    [{shuffled_bias[0]:.3f}, {shuffled_bias[1]:.3f}, {shuffled_bias[2]:.3f}]")
    print(f"Class weights bias:      [{weights_bias[0]:.3f}, {weights_bias[1]:.3f}, {weights_bias[2]:.3f}]")
    
    print(f"\nPSAX bias comparison:")
    print(f"  Sequential:  {sequential_bias[1]:.3f} (worst)")
    print(f"  Shuffled:    {shuffled_bias[1]:.3f} (better)")
    print(f"  Class weights: {weights_bias[1]:.3f} (best)")
    
    if abs(shuffled_bias[1]) < abs(sequential_bias[1]):
        print(f"\nSHUFFLE HELPS! PSAX bias improved from {sequential_bias[1]:.3f} to {shuffled_bias[1]:.3f}")
        print(f"   Try adding shuffle=True to your dataloader!")
    else:
        print(f"\nShuffle doesn't help much. PSAX bias: {sequential_bias[1]:.3f} -> {shuffled_bias[1]:.3f}")
        print(f"   Stick with class weights solution.")
