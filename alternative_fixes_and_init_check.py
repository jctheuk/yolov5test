"""
Alternative methods to fix PSAX bias bug and check model initialization

Methods to fix class imbalance bias:
1. Class weights (already implemented)
2. Focal Loss - reduces loss for easy examples
3. Label smoothing - prevents overconfidence
4. Bias initialization with class priors
5. Balanced sampling
6. Threshold adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append('yolov5c')

def check_model_initialization():
    """Check how the model is initialized"""
    print("=" * 60)
    print("CHECKING MODEL INITIALIZATION")
    print("=" * 60)
    
    # Check if there's a trained model to examine
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    if Path(model_path).exists():
        print(f"\nFound trained model: {model_path}")
        
        # Load model
        model = torch.load(model_path, map_location='cpu')
        
        # Find classification head
        for name, module in model['model'].named_modules():
            if 'linear' in name.lower() and isinstance(module, nn.Linear):
                if module.weight.shape[0] == 3:  # Classification head
                    print(f"\nClassification head found: {name}")
                    print(f"  Weight shape: {module.weight.shape}")
                    print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'None'}")
                    
                    # Check initialization values
                    print(f"\nCurrent values (after training):")
                    print(f"  Weight mean: {module.weight.mean():.6f}")
                    print(f"  Weight std: {module.weight.std():.6f}")
                    print(f"  Bias: {module.bias.data if module.bias is not None else 'None'}")
                    
                    # Check if bias is initialized correctly
                    if module.bias is not None:
                        print(f"\nBias analysis:")
                        for i, class_name in enumerate(['A4C', 'PSAX', 'PLAX']):
                            bias_val = module.bias[i].item()
                            print(f"  {class_name}: {bias_val:.6f}")
                            if abs(bias_val) > 0.1:
                                print(f"    WARNING: {class_name} bias is large!")
        
        # Check epoch and training info
        print(f"\nTraining info:")
        print(f"  Epoch: {model.get('epoch', 'Unknown')}")
        print(f"  Best fitness: {model.get('best_fitness', 'Unknown')}")
        
    else:
        print(f"\nNo trained model found at {model_path}")
        print("Let's check a fresh model initialization...")
        
        # Create a fresh model and check initialization
        from models.yolo import Model
        model = Model('yolov5c/models/yolov5sc_classify_backbone.yaml', ch=3, nc=4, anchors=None)
        
        # Find classification head
        for name, module in model.named_modules():
            if 'linear' in name.lower() and isinstance(module, nn.Linear):
                if module.weight.shape[0] == 3:  # Classification head
                    print(f"\nFresh model classification head: {name}")
                    print(f"  Weight shape: {module.weight.shape}")
                    print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'None'}")
                    
                    print(f"\nInitialization values:")
                    print(f"  Weight mean: {module.weight.mean():.6f}")
                    print(f"  Weight std: {module.weight.std():.6f}")
                    print(f"  Bias: {module.bias.data if module.bias is not None else 'None'}")

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    Focal Loss implementation to handle class imbalance
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Target class indices [batch_size]
        alpha: Class weighting factors [num_classes]
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    # Compute cross entropy
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Compute probabilities
    p_t = torch.exp(-ce_loss)
    
    # Apply focal weighting
    focal_weight = (1 - p_t) ** gamma
    
    # Apply class weights if provided
    if alpha is not None:
        alpha_t = alpha[targets]
        focal_weight = alpha_t * focal_weight
    
    focal_loss = focal_weight * ce_loss
    return focal_loss.mean()

def test_focal_loss():
    """Test focal loss as alternative to class weights"""
    print("\n" + "=" * 60)
    print("TESTING FOCAL LOSS ALTERNATIVE")
    print("=" * 60)
    
    # Your class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    # Calculate alpha weights for focal loss
    alpha = total / (len(class_counts) * class_counts)
    alpha = alpha / alpha.mean()  # Normalize
    
    print(f"Class distribution: {class_counts}")
    print(f"Class probabilities: {class_probs}")
    print(f"Focal loss alpha weights: {alpha}")
    
    # Test focal loss
    torch.manual_seed(42)
    num_classes = 3
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    nn.init.constant_(linear.bias, 0.0)
    
    # Convert alpha to tensor
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    
    print(f"\nInitial bias: {linear.bias.data}")
    
    # Train with focal loss
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
        
        # Forward pass with focal loss
        logits = linear(features)
        loss = focal_loss(logits, labels, alpha=alpha_tensor, gamma=2.0)
        
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
    print(f"\nFinal bias with focal loss: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")

def bias_initialization_with_priors():
    """Initialize bias with class priors instead of zero"""
    print("\n" + "=" * 60)
    print("TESTING BIAS INITIALIZATION WITH PRIORS")
    print("=" * 60)
    
    # Your class distribution
    class_counts = np.array([324, 218, 455])  # A4C, PSAX, PLAX
    total = class_counts.sum()
    class_probs = class_counts / total
    
    print(f"Class probabilities: {class_probs}")
    
    # Calculate bias initialization with priors
    # bias = log(prior) - log(1/num_classes)
    num_classes = 3
    bias_init = np.log(class_probs) - np.log(1.0/num_classes)
    
    print(f"Calculated bias initialization: {bias_init}")
    
    # Test training with prior initialization
    torch.manual_seed(42)
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    
    # Initialize bias with priors instead of zero
    nn.init.constant_(linear.bias, 0.0)  # Start with zero
    with torch.no_grad():
        linear.bias.data = torch.tensor(bias_init, dtype=torch.float32)
    
    print(f"Initial bias with priors: {linear.bias.data}")
    
    # Train normally
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
    print(f"\nFinal bias with prior initialization: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")

def balanced_sampling_approach():
    """Test balanced sampling to fix class imbalance"""
    print("\n" + "=" * 60)
    print("TESTING BALANCED SAMPLING APPROACH")
    print("=" * 60)
    
    # Simulate balanced sampling (equal samples per class)
    balanced_probs = [1/3, 1/3, 1/3]  # Equal probability
    
    print(f"Balanced sampling probabilities: {balanced_probs}")
    
    torch.manual_seed(42)
    num_classes = 3
    feature_dim = 1280
    linear = nn.Linear(feature_dim, num_classes)
    nn.init.constant_(linear.bias, 0.0)
    
    print(f"Initial bias: {linear.bias.data}")
    
    # Train with balanced sampling
    lr = 0.01
    for epoch in range(100):
        # Simulate batch with balanced distribution
        batch_size = 30  # Must be divisible by 3
        class_counts_batch = [batch_size // 3] * 3  # Equal counts
        
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
    print(f"\nFinal bias with balanced sampling: [{final_bias[0]:.3f}, {final_bias[1]:.3f}, {final_bias[2]:.3f}]")

if __name__ == "__main__":
    check_model_initialization()
    test_focal_loss()
    bias_initialization_with_priors()
    balanced_sampling_approach()
    
    print("\n" + "=" * 60)
    print("ALTERNATIVE SOLUTIONS SUMMARY")
    print("=" * 60)
    print("\n1. Class weights (implemented): [1.026, 1.524, 0.730]")
    print("2. Focal loss: Reduces loss for easy examples")
    print("3. Bias initialization with priors: Start with log(prior) bias")
    print("4. Balanced sampling: Equal samples per class")
    print("5. Label smoothing: Prevents overconfidence")
    print("\nTry different approaches to see which works best!")
