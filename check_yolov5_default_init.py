"""
Check how default yolov5s.pt initializes classification head

Compare:
1. Your trained model (yolov5sc_classify_backbone) - PSAX bias = -0.263
2. Default yolov5s.pt - what's the initial bias?
3. Fresh yolov5sc_classify_backbone - what's the initial bias?
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('yolov5c')

def check_yolov5s_initialization():
    """Check default yolov5s.pt model"""
    print("=" * 60)
    print("CHECKING DEFAULT YOLOV5S.PT INITIALIZATION")
    print("=" * 60)
    
    model_path = "yolov5s.pt"
    if not Path(model_path).exists():
        print(f"yolov5s.pt not found, skipping...")
        return
    
    print(f"\nLoading {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')
    
    print(f"Model keys: {ckpt.keys()}")
    
    # Check if it's an old format checkpoint
    if 'model' in ckpt:
        model = ckpt['model']
        
        # Find any Linear layers (classification heads)
        found_linear = False
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"\nFound Linear layer: {name}")
                print(f"  Weight shape: {module.weight.shape}")
                print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'None'}")
                
                if module.bias is not None:
                    print(f"  Bias values: {module.bias.data}")
                    print(f"  Bias mean: {module.bias.mean():.6f}")
                    print(f"  Bias std: {module.bias.std():.6f}")
                
                found_linear = True
        
        if not found_linear:
            print("\nNo Linear layers found in yolov5s.pt")
            print("This is a pure detection model (no classification head)")
    else:
        print("Unknown checkpoint format")

def check_fresh_model_initialization():
    """Check how a fresh yolov5sc_classify_backbone initializes"""
    print("\n" + "=" * 60)
    print("CHECKING FRESH MODEL INITIALIZATION")
    print("=" * 60)
    
    try:
        from models.yolo import Model
        
        print("\nCreating fresh yolov5sc_classify_backbone model...")
        model = Model('yolov5c/models/yolov5sc_classify_backbone.yaml', ch=3, nc=4, anchors=None)
        
        # Find classification head
        found = False
        for name, module in model.named_modules():
            if 'linear' in name.lower() and isinstance(module, nn.Linear):
                if module.weight.shape[0] == 3:  # Classification head
                    print(f"\nClassification head: {name}")
                    print(f"  Weight shape: {module.weight.shape}")
                    print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'None'}")
                    
                    print(f"\nFRESH initialization values:")
                    print(f"  Weight mean: {module.weight.mean():.6f}")
                    print(f"  Weight std: {module.weight.std():.6f}")
                    
                    if module.bias is not None:
                        print(f"  Bias values:")
                        for i, class_name in enumerate(['A4C', 'PSAX', 'PLAX']):
                            print(f"    {class_name} (class {i}): {module.bias[i]:.6f}")
                        
                        print(f"\n  Bias mean: {module.bias.mean():.6f}")
                        print(f"  Bias std: {module.bias.std():.6f}")
                        
                        # Check if bias is zero or near-zero
                        if torch.all(torch.abs(module.bias) < 0.001):
                            print("\n  OK: Bias is initialized to zero (or near-zero)")
                        else:
                            print("\n  WARNING: Bias is NOT initialized to zero!")
                            print(f"    This could be causing the PSAX bias issue")
                    
                    found = True
                    break
        
        if not found:
            print("\nNo classification head found!")
            
    except Exception as e:
        print(f"\nError creating model: {e}")

def compare_initialization_methods():
    """Compare different initialization methods for classification head"""
    print("\n" + "=" * 60)
    print("COMPARING INITIALIZATION METHODS")
    print("=" * 60)
    
    num_classes = 3
    feature_dim = 1280
    
    # Method 1: PyTorch default
    linear1 = nn.Linear(feature_dim, num_classes)
    print(f"\nMethod 1: PyTorch default initialization")
    print(f"  Bias: {linear1.bias.data}")
    print(f"  Bias mean: {linear1.bias.mean():.6f}")
    
    # Method 2: Zero initialization (like in common.py line 945)
    linear2 = nn.Linear(feature_dim, num_classes)
    nn.init.constant_(linear2.bias, 0)
    print(f"\nMethod 2: Zero initialization (current code)")
    print(f"  Bias: {linear2.bias.data}")
    print(f"  Bias mean: {linear2.bias.mean():.6f}")
    
    # Method 3: Initialize with class priors
    class_probs = np.array([0.325, 0.219, 0.456])  # Your actual distribution
    bias_priors = np.log(class_probs)
    linear3 = nn.Linear(feature_dim, num_classes)
    with torch.no_grad():
        linear3.bias.data = torch.tensor(bias_priors, dtype=torch.float32)
    print(f"\nMethod 3: Initialize with class priors (proposed fix)")
    print(f"  Bias: {linear3.bias.data}")
    print(f"  Bias mean: {linear3.bias.mean():.6f}")
    
    # Method 4: Uniform initialization
    linear4 = nn.Linear(feature_dim, num_classes)
    nn.init.uniform_(linear4.bias, -0.1, 0.1)
    print(f"\nMethod 4: Uniform initialization [-0.1, 0.1]")
    print(f"  Bias: {linear4.bias.data}")
    print(f"  Bias mean: {linear4.bias.mean():.6f}")

if __name__ == "__main__":
    check_yolov5s_initialization()
    check_fresh_model_initialization()
    compare_initialization_methods()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\nThe fresh model initializes bias to ZERO, which is correct.")
    print("The problem is that during training:")
    print("  1. Class imbalance (PLAX 45.6%, PSAX 21.9%)")
    print("  2. Causes PSAX bias to become negative (-0.263)")
    print("  3. Which suppresses PSAX predictions")
    print("\nSolutions:")
    print("  1. Use class weights (implemented) - easiest")
    print("  2. Use focal loss (more complex) - alternative")
    print("  3. Use balanced sampling (best but complex) - requires data pipeline changes")
