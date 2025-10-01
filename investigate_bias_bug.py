"""
Investigate bias initialization and gradient flow bug

The PSAX bias became -0.263 during training (started at 0).
This is the root cause of 9% PSAX recall.

Need to find:
1. Why PSAX bias becomes negative during training
2. Is there a bug in bias initialization?
3. Is there a bug in gradient updates for class 1?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append('yolov5c')

def check_initialization_code():
    """Check the bias initialization code"""
    print("=" * 60)
    print("CHECKING BIAS INITIALIZATION CODE")
    print("=" * 60)
    
    # Read the classification head initialization
    with open('yolov5c/models/common.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the initialization section
    for i, line in enumerate(lines):
        if 'def _initialize_weights' in line:
            print(f"\nFound initialization function at line {i+1}:")
            # Print next 20 lines
            for j in range(i, min(i+20, len(lines))):
                print(f"{j+1:3d}: {lines[j].rstrip()}")
            break
    
    # Check if there's any class-specific initialization
    print("\nSearching for class-specific initialization...")
    found_class_specific = False
    for i, line in enumerate(lines):
        if 'class' in line.lower() and ('init' in line.lower() or 'bias' in line.lower()):
            print(f"Line {i+1}: {line.rstrip()}")
            found_class_specific = True
    
    if not found_class_specific:
        print("No class-specific initialization found")

def simulate_bias_evolution():
    """Simulate how bias evolves during training"""
    print("\n" + "=" * 60)
    print("SIMULATING BIAS EVOLUTION")
    print("=" * 60)
    
    # Simulate the training scenario
    torch.manual_seed(42)  # For reproducibility
    
    # Create a simple classification head
    num_classes = 3
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    
    # Initialize bias to zero (like the code does)
    nn.init.constant_(linear.bias, 0.0)
    print(f"\nInitial bias: {linear.bias.data}")
    
    # Simulate different training scenarios
    scenarios = [
        ("Balanced data", [0.33, 0.33, 0.34]),  # Equal class distribution
        ("Actual data", [0.325, 0.219, 0.456]),  # Your actual distribution
        ("Extreme imbalance", [0.1, 0.05, 0.85]),  # Very imbalanced
    ]
    
    for scenario_name, class_probs in scenarios:
        print(f"\n{scenario_name} scenario:")
        print(f"Class probabilities: {class_probs}")
        
        # Reset bias
        nn.init.constant_(linear.bias, 0.0)
        
        # Simulate training with this distribution
        lr = 0.01
        for epoch in range(100):
            # Simulate batch with this class distribution
            batch_size = 32
            class_counts = np.random.multinomial(batch_size, class_probs)
            
            # Create mock features and labels
            features = torch.randn(batch_size, feature_dim)
            labels = []
            for class_idx, count in enumerate(class_counts):
                labels.extend([class_idx] * count)
            labels = torch.tensor(labels[:batch_size])
            
            # Forward pass
            logits = linear(features)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights (simple SGD)
            with torch.no_grad():
                linear.bias -= lr * linear.bias.grad
                linear.weight -= lr * linear.weight.grad
                linear.bias.grad.zero_()
                linear.weight.grad.zero_()
            
            # Print bias every 20 epochs
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}: bias = [{linear.bias[0]:.3f}, {linear.bias[1]:.3f}, {linear.bias[2]:.3f}]")
        
        final_bias = linear.bias.data
        print(f"  Final bias: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")
        
        # Check if PSAX (class 1) bias is negative
        if final_bias[1] < -0.1:
            print(f"  ⚠️  PSAX bias is strongly negative: {final_bias[1]:.3f}")
            print(f"      This matches your observed bug!")

def check_label_smoothing_effect():
    """Check if label smoothing affects class 1 differently"""
    print("\n" + "=" * 60)
    print("CHECKING LABEL SMOOTHING EFFECT")
    print("=" * 60)
    
    # Test with and without label smoothing
    num_classes = 3
    
    # Create mock logits and labels
    logits = torch.randn(4, num_classes, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 2])  # Include PSAX labels
    
    print(f"\nTest labels: {labels}")
    print(f"PSAX samples at indices: {torch.where(labels == 1)[0].tolist()}")
    
    # Test 1: No label smoothing
    loss_no_smooth = F.cross_entropy(logits, labels)
    loss_no_smooth.backward(retain_graph=True)
    grad_no_smooth = logits.grad.clone()
    logits.grad.zero_()
    
    # Test 2: With label smoothing (0.1 like in your config)
    loss_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
    loss_smooth.backward(retain_graph=True)
    grad_smooth = logits.grad.clone()
    
    print(f"\nGradient comparison:")
    print(f"  No smoothing: {grad_no_smooth.mean(dim=0)}")
    print(f"  With smoothing: {grad_smooth.mean(dim=0)}")
    
    # Check if PSAX (class 1) gradients are different
    psax_indices = torch.where(labels == 1)[0]
    if len(psax_indices) > 0:
        psax_grad_no_smooth = grad_no_smooth[psax_indices]
        psax_grad_smooth = grad_smooth[psax_indices]
        
        print(f"\nPSAX sample gradients:")
        print(f"  No smoothing: {psax_grad_no_smooth.mean(dim=0)}")
        print(f"  With smoothing: {psax_grad_smooth.mean(dim=0)}")
        
        # Check if smoothing affects PSAX differently
        diff = psax_grad_smooth - psax_grad_no_smooth
        if abs(diff.mean(dim=0)[1]) > 0.01:  # Class 1 gradient difference
            print(f"  ⚠️  Label smoothing affects PSAX gradients differently!")

def check_cross_entropy_bug():
    """Check if there's a bug in cross entropy for class 1"""
    print("\n" + "=" * 60)
    print("CHECKING CROSS ENTROPY FOR CLASS 1")
    print("=" * 60)
    
    # Test cross entropy with different class labels
    num_classes = 3
    batch_size = 6
    
    # Create logits
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    
    # Test with each class
    for class_idx, class_name in enumerate(['A4C', 'PSAX', 'PLAX']):
        labels = torch.full((batch_size,), class_idx, dtype=torch.long)
        
        # Forward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward(retain_graph=True)
        
        # Check gradients
        grad_mean = logits.grad.mean(dim=0)
        print(f"\n{class_name} (class {class_idx}) gradients:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients: {grad_mean}")
        
        # Check if class 1 has unusual gradient pattern
        if class_idx == 1:  # PSAX
            if abs(grad_mean[1]) < abs(grad_mean[0]) * 0.5:
                print(f"  ⚠️  PSAX self-gradient is unusually small!")
            if grad_mean[1] > 0:
                print(f"  ⚠️  PSAX self-gradient is positive (should be negative)!")
        
        logits.grad.zero_()

if __name__ == "__main__":
    check_initialization_code()
    simulate_bias_evolution()
    check_label_smoothing_effect()
    check_cross_entropy_bug()
    
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print("\nIf any test shows unusual behavior for PSAX (class 1),")
    print("that's likely the source of the bias bug.")
