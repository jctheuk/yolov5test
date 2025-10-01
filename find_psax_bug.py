"""
Systematic investigation to find the PSAX (class 1) bug

Evidence:
- PSAX has 21.9% of training data
- PSAX only gets 9% recall (should be ~22%)
- Confusion matrix shows PSAX is predicted as A4C (57.6%) or PLAX (33.3%)

Possible bugs:
1. Model initialization biases class 1 weights to be weaker
2. Loss function has a bug for class 1
3. Gradient updates are not applying properly to class 1
4. There's a hardcoded index swap somewhere
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append('yolov5c')

from models.experimental import attempt_load
from utils.general import check_dataset
from utils.dataloaders import create_dataloader

def check_model_classification_head_weights():
    """Check if classification head weights are biased"""
    print("=" * 60)
    print("CHECKING CLASSIFICATION HEAD WEIGHTS")
    print("=" * 60)
    
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    model = attempt_load(model_path, device='cpu', inplace=False, fuse=False)
    
    # Find classification head
    print("\nSearching for classification head...")
    for name, module in model.named_modules():
        if 'linear' in name.lower() or 'classifier' in name.lower():
            if isinstance(module, nn.Linear):
                print(f"\nFound classification head: {name}")
                print(f"  Weight shape: {module.weight.shape}")
                print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'None'}")
                
                # Check weights for each class
                if module.weight.shape[0] == 3:  # 3 classes
                    for i, class_name in enumerate(['A4C', 'PSAX', 'PLAX']):
                        weights = module.weight[i, :]
                        print(f"\n  Class {i} ({class_name}) weights:")
                        print(f"    Mean: {weights.mean():.6f}")
                        print(f"    Std: {weights.std():.6f}")
                        print(f"    Min: {weights.min():.6f}")
                        print(f"    Max: {weights.max():.6f}")
                        print(f"    Norm: {weights.norm():.6f}")
                    
                    # Check if class 1 (PSAX) has significantly different weights
                    norm_0 = module.weight[0, :].norm()
                    norm_1 = module.weight[1, :].norm()
                    norm_2 = module.weight[2, :].norm()
                    
                    print(f"\n  Weight norm comparison:")
                    print(f"    A4C (0):  {norm_0:.4f}")
                    print(f"    PSAX (1): {norm_1:.4f}")
                    print(f"    PLAX (2): {norm_2:.4f}")
                    
                    if norm_1 < norm_0 * 0.5 or norm_1 < norm_2 * 0.5:
                        print(f"\n    CRITICAL: PSAX weights are significantly weaker!")
                        print(f"      This explains why PSAX is underpredicted")
                    
                    # Check bias
                    if module.bias is not None:
                        print(f"\n  Bias values:")
                        for i, class_name in enumerate(['A4C', 'PSAX', 'PLAX']):
                            print(f"    {class_name}: {module.bias[i]:.6f}")
                        
                        if abs(module.bias[1]) > abs(module.bias[0]) * 2 or abs(module.bias[1]) > abs(module.bias[2]) * 2:
                            print(f"\n    WARNING: PSAX bias is unusual!")

def check_loss_gradient_for_psax():
    """Check if gradients flow properly for PSAX samples"""
    print("\n" + "=" * 60)
    print("CHECKING GRADIENT FLOW FOR PSAX")
    print("=" * 60)
    
    # Create mock data with PSAX labels
    batch_size = 4
    num_classes = 3
    
    # Mock classification output (logits)
    classification_output = torch.randn(batch_size, num_classes, requires_grad=True)
    
    # Test 1: All PSAX labels
    psax_labels = torch.ones(batch_size, dtype=torch.long)  # All class 1
    
    print("\nTest 1: Computing loss for all-PSAX batch...")
    loss = nn.CrossEntropyLoss()(classification_output, psax_labels)
    print(f"  Loss: {loss.item():.4f}")
    
    loss.backward()
    print(f"  Gradient norm: {classification_output.grad.norm():.4f}")
    
    # Check if gradients are similar for all classes
    grad_sum_per_class = classification_output.grad.sum(dim=0)
    print(f"\n  Gradient sum per class:")
    for i, name in enumerate(['A4C', 'PSAX', 'PLAX']):
        print(f"    {name}: {grad_sum_per_class[i]:.6f}")
    
    if abs(grad_sum_per_class[1]) < abs(grad_sum_per_class[0]) * 0.5:
        print(f"\n    WARNING: PSAX gradients are significantly smaller!")
        print(f"      This could prevent PSAX from learning properly")

if __name__ == "__main__":
    check_model_classification_head_weights()
    check_loss_gradient_for_psax()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nPotential bugs to investigate further:")
    print("  1. If PSAX weights are weaker -> model initialization bug")
    print("  2. If PSAX gradients are smaller -> loss function bug")
    print("  3. If neither -> check training loop for PSAX-specific issues")

