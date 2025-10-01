"""
Test fix for PSAX bias bug

The issue is class imbalance causing PSAX bias to become negative.
We can fix this by:
1. Using class weights in loss function
2. Resetting bias periodically
3. Using focal loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_class_weighted_loss():
    """Test if class weights fix the bias issue"""
    print("=" * 60)
    print("TESTING CLASS WEIGHTED LOSS FIX")
    print("=" * 60)
    
    # Your actual class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    # Calculate inverse class weights (standard approach)
    class_weights = total / (len(class_counts) * class_counts)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    print(f"Class weights: {class_weights}")
    
    # Simulate training with class weights
    torch.manual_seed(42)
    num_classes = 3
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    nn.init.constant_(linear.bias, 0.0)
    
    # Create weighted loss function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    
    print(f"\nInitial bias: {linear.bias.data}")
    
    # Train with class weights
    lr = 0.01
    for epoch in range(100):
        # Simulate batch with actual distribution
        batch_size = 32
        class_counts_batch = np.random.multinomial(batch_size, class_probs)
        
        features = torch.randn(batch_size, feature_dim)
        labels = []
        for class_idx, count in enumerate(class_counts_batch):
            labels.extend([class_idx] * count)
        labels = torch.tensor(labels[:batch_size])
        
        # Forward pass with weighted loss
        logits = linear(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        with torch.no_grad():
            linear.bias -= lr * linear.bias.grad
            linear.weight -= lr * linear.weight.grad
            linear.bias.grad.zero_()
            linear.weight.grad.zero_()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}]")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias with class weights: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
    
    # Check if PSAX bias is still negative
    if final_bias[1] < -0.05:
        print(f"  PSAX bias is still negative: {final_bias[1]:.3f}")
        print(f"  Class weights alone may not be enough")
    else:
        print(f"  ✅ PSAX bias is reasonable: {final_bias[1]:.3f}")
        print(f"  Class weights fixed the issue!")

def test_bias_reset_approach():
    """Test resetting bias to zero periodically"""
    print("\n" + "=" * 60)
    print("TESTING BIAS RESET APPROACH")
    print("=" * 60)
    
    torch.manual_seed(42)
    num_classes = 3
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    nn.init.constant_(linear.bias, 0.0)
    
    # Your actual class distribution
    class_probs = [0.325, 0.219, 0.456]
    
    lr = 0.01
    for epoch in range(100):
        # Reset bias every 25 epochs
        if epoch > 0 and epoch % 25 == 0:
            nn.init.constant_(linear.bias, 0.0)
            print(f"  Epoch {epoch:3d}: Reset bias to zero")
        
        # Simulate training
        batch_size = 32
        class_counts_batch = np.random.multinomial(batch_size, class_probs)
        
        features = torch.randn(batch_size, feature_dim)
        labels = []
        for class_idx, count in enumerate(class_counts_batch):
            labels.extend([class_idx] * count)
        labels = torch.tensor(labels[:batch_size])
        
        # Forward pass
        logits = linear(features)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        with torch.no_grad():
            linear.bias -= lr * linear.bias.grad
            linear.weight -= lr * linear.weight.grad
            linear.bias.grad.zero_()
            linear.weight.grad.zero_()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}]")
    
    final_bias = linear.bias.data
    print(f"\nFinal bias with periodic reset: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")

def create_fixed_hyperparameters():
    """Create hyperparameters file with class weights"""
    print("\n" + "=" * 60)
    print("CREATING FIXED HYPERPARAMETERS")
    print("=" * 60)
    
    # Calculate class weights
    class_counts = np.array([324, 218, 455])  # Your actual counts
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    
    # Normalize weights to have mean = 1
    class_weights = class_weights / class_weights.mean()
    
    print(f"Calculated class weights: {class_weights}")
    
    # Create hyperparameters file
    content = f"""# YOLOv5 Classification-Only Hyperparameters
# FIXED for PSAX bias bug using class weights

# Learning rate settings
lr0: 0.001
lrf: 0.1

# Optimizer settings
momentum: 0.937
weight_decay: 0.0005

# Warmup settings
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Detection loss weights (DISABLED)
box: 0.0
cls: 0.0
obj: 0.0

# Additional detection parameters
cls_pw: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0

# Classification-specific parameters
cls_task: 1.0
label_smoothing: 0.1

# Class weights to fix PSAX bias bug
class_weights: [{class_weights[0]:.3f}, {class_weights[1]:.3f}, {class_weights[2]:.3f}]

# Data augmentation (DISABLED per project rules)
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
"""
    
    with open('psax_bias_fix_hyp.yaml', 'w') as f:
        f.write(content)
    
    print(f"\nCreated 'psax_bias_fix_hyp.yaml' with class weights")
    print(f"Class weights: A4C={class_weights[0]:.3f}, PSAX={class_weights[1]:.3f}, PLAX={class_weights[2]:.3f}")

if __name__ == "__main__":
    test_class_weighted_loss()
    test_bias_reset_approach()
    create_fixed_hyperparameters()
    
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print("\nThe PSAX bias bug is caused by class imbalance.")
    print("Solutions:")
    print("1. Use class weights in loss function")
    print("2. Reset bias periodically during training")
    print("3. Use focal loss instead of cross entropy")
    print("\nTry training with 'psax_bias_fix_hyp.yaml' and class weights!")
