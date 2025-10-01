"""
Trace exactly where and how the classification head bias becomes [0, 0, 0]

Steps:
1. PyTorch creates nn.Linear with default initialization (small random bias)
2. Check if model loading from yolov5s.pt resets it
3. Check if there's a reset_parameters call
4. Check if the bias is overwritten during model creation
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append('yolov5c')

def step1_pytorch_default():
    """Step 1: Check PyTorch's default Linear initialization"""
    print("=" * 60)
    print("STEP 1: PYTORCH DEFAULT LINEAR INITIALIZATION")
    print("=" * 60)
    
    # Create multiple Linear layers to see the pattern
    print("\nCreating 5 fresh nn.Linear(1280, 3) layers:")
    for i in range(5):
        linear = nn.Linear(1280, 3)
        print(f"  Trial {i+1}: bias = {linear.bias.data}")
    
    print("\nConclusion: PyTorch defaults to SMALL RANDOM bias, NOT zero!")

def step2_model_yaml_creation():
    """Step 2: Check what happens when creating model from yaml"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL CREATION FROM YAML")
    print("=" * 60)
    
    try:
        from models.yolo import Model
        
        # Create model without loading any weights
        print("\nCreating model from yolov5sc_classify_backbone.yaml (no pretrained weights)...")
        print("This requires fixing the yaml first (num_cls -> 3)...")
        
        # Read yaml and check for num_cls
        import yaml
        with open('yolov5c/models/yolov5sc_classify_backbone.yaml', 'r') as f:
            yaml_content = f.read()
        
        if 'num_cls' in yaml_content:
            print("\nFound 'num_cls' in yaml - this needs to be resolved at model creation")
            print("The model creation process should replace 'num_cls' with actual value (3)")
        
    except Exception as e:
        print(f"Error: {e}")

def step3_weight_loading():
    """Step 3: Check what happens during weight loading from yolov5s.pt"""
    print("\n" + "=" * 60)
    print("STEP 3: WEIGHT LOADING FROM YOLOV5S.PT")
    print("=" * 60)
    
    # Check train_classification_task.py weight loading logic
    print("\nIn train_classification_task.py lines 693-738:")
    print("  1. Creates new model from yaml (classification head gets PyTorch default bias)")
    print("  2. Loads yolov5s.pt checkpoint")
    print("  3. Calls model.load_state_dict(csd, strict=False)")
    print("\nThe key question: Does yolov5s.pt have weights for classification head?")
    
    # Check yolov5s.pt structure
    if Path('yolov5s.pt').exists():
        ckpt = torch.load('yolov5s.pt', map_location='cpu')
        state_dict = ckpt['model'].float().state_dict()
        
        print(f"\nChecking yolov5s.pt state_dict for classification head...")
        classification_keys = [k for k in state_dict.keys() if 'linear' in k.lower() or 'classifier' in k.lower()]
        
        if classification_keys:
            print(f"  Found classification keys: {classification_keys}")
            for key in classification_keys:
                print(f"    {key}: shape {state_dict[key].shape}")
        else:
            print(f"  NO classification head in yolov5s.pt")
            print(f"  Total keys: {len(state_dict)}")
            print(f"  Sample keys: {list(state_dict.keys())[:5]}")
            
            print("\n  IMPORTANT: yolov5s.pt doesn't have classification head weights!")
            print("  So the classification head keeps its PyTorch default initialization.")
            print("  Bias should be SMALL RANDOM, not [0, 0, 0]!")

def step4_check_if_something_resets_bias():
    """Step 4: Check if something else resets the bias to zero"""
    print("\n" + "=" * 60)
    print("STEP 4: SEARCHING FOR BIAS RESET CODE")
    print("=" * 60)
    
    print("\nSearching for code that might reset classification bias to zero...")
    
    # Check train_classification_task.py for bias reset
    with open('train_classification_task.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for patterns that might reset bias
    patterns = [
        'bias.data.zero_',
        'bias.fill_(0)',
        'bias.data = 0',
        'init.constant_(bias, 0)',
        'bias = torch.zeros',
    ]
    
    found_any = False
    for pattern in patterns:
        if pattern in content:
            print(f"  Found pattern: '{pattern}'")
            found_any = True
    
    if not found_any:
        print("  No explicit bias reset code found in train_classification_task.py")
    
    # Check if model.half().float() might affect bias
    if 'model.half().float()' in content:
        print("\n  Found: model.half().float() - this converts precision")
        print("  This should NOT reset bias to zero, but let's verify...")
        
        # Test if half().float() resets bias
        linear = nn.Linear(1280, 3)
        original_bias = linear.bias.data.clone()
        linear = linear.half().float()
        new_bias = linear.bias.data
        
        print(f"\n  Testing model.half().float() effect on bias:")
        print(f"    Original: {original_bias}")
        print(f"    After half().float(): {new_bias}")
        print(f"    Difference: {(new_bias - original_bias).abs().max():.6f}")
        
        if torch.allclose(original_bias, new_bias, atol=1e-4):
            print(f"    OK: half().float() preserves bias values")
        else:
            print(f"    WARNING: half().float() changes bias values!")

if __name__ == "__main__":
    step1_pytorch_default()
    step2_model_yaml_creation()
    step3_weight_loading()
    step4_check_if_something_resets_bias()
    
    print("\n" + "=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)
    print("\nThe classification head bias should be SMALL RANDOM, not [0, 0, 0]")
    print("If you see [0, 0, 0], something in the code is resetting it.")
    print("This could be:")
    print("  1. Weight loading from a checkpoint with zero bias")
    print("  2. Explicit bias reset in training code")
    print("  3. A side effect of model.half().float() or similar operations")
