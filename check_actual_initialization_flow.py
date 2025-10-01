"""
Check the ACTUAL initialization flow in your training

When you run:
python train_classification_task.py --weights yolov5s.pt ...

What actually happens to the classification head bias?
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
sys.path.append('yolov5c')

from models.yolo import Model
from utils.general import intersect_dicts

def simulate_actual_training_initialization():
    """Simulate what happens in train_classification_task.py lines 686-740"""
    print("=" * 60)
    print("SIMULATING ACTUAL TRAINING INITIALIZATION")
    print("=" * 60)
    
    # Load data.yaml to get nc
    with open('regurgitationV1/data.yaml', 'r') as f:
        data_dict = yaml.safe_load(f)
    
    nc = data_dict['nc']  # Number of detection classes
    num_cls = data_dict['num_cls']  # Number of classification classes
    
    print(f"\nFrom data.yaml:")
    print(f"  Detection classes (nc): {nc}")
    print(f"  Classification classes (num_cls): {num_cls}")
    
    # Step 1: Load checkpoint
    weights = 'yolov5s.pt'
    print(f"\nStep 1: Loading weights from {weights}...")
    ckpt = torch.load(weights, map_location='cpu')
    
    # Step 2: Create new model from yaml (line 693)
    print(f"\nStep 2: Creating model from yaml...")
    try:
        # This is what line 693 does
        model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=None)
        print(f"  Model created successfully")
        
        # Find classification head to check initial bias
        for name, module in model.named_modules():
            if 'linear' in name.lower() and isinstance(module, nn.Linear):
                if module.weight.shape[0] == num_cls:
                    print(f"\n  Classification head after creation: {name}")
                    print(f"    Initial bias: {module.bias.data}")
                    initial_bias = module.bias.data.clone()
                    
                    # Step 3: Load state dict (line 735-737)
                    print(f"\nStep 3: Loading state dict from yolov5s.pt...")
                    csd = ckpt['model'].float().state_dict()
                    
                    # Check if classification head key exists in checkpoint
                    linear_keys = [k for k in csd.keys() if name in k or 'linear' in k.lower()]
                    if linear_keys:
                        print(f"    Found classification head in checkpoint: {linear_keys}")
                    else:
                        print(f"    NO classification head in checkpoint")
                        print(f"    Classification head will keep PyTorch default bias!")
                    
                    # intersect_dicts (line 736)
                    exclude = ['anchor']
                    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
                    
                    # Check if classification head is in intersected dict
                    linear_keys_intersect = [k for k in csd.keys() if name in k or ('linear' in k.lower() and 'bias' in k.lower())]
                    if linear_keys_intersect:
                        print(f"    Classification head in intersected dict: {linear_keys_intersect}")
                        for key in linear_keys_intersect:
                            if 'bias' in key.lower():
                                print(f"      {key}: {csd[key]}")
                    else:
                        print(f"    NO classification head in intersected dict")
                        print(f"    Classification head bias will NOT be overwritten!")
                    
                    # Load state dict (line 737)
                    model.load_state_dict(csd, strict=False)
                    
                    # Check bias after loading
                    print(f"\n  After load_state_dict:")
                    print(f"    Final bias: {module.bias.data}")
                    
                    # Compare
                    if torch.allclose(module.bias.data, initial_bias, atol=1e-6):
                        print(f"    ✅ Bias UNCHANGED (as expected - no classification weights in yolov5s.pt)")
                    else:
                        print(f"    WARNING: Bias CHANGED!")
                        print(f"    Initial: {initial_bias}")
                        print(f"    Final: {module.bias.data}")
                    
                    break
    
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_actual_training_initialization()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\nIf bias stays as small random values after initialization,")
    print("then something ELSE is causing it to become [0, 0, 0] or evolve to [-0.263]")
    print("\nPossible causes:")
    print("  1. You're loading a checkpoint (not yolov5s.pt) that already has trained bias")
    print("  2. There's code somewhere that resets classification head bias")
    print("  3. The bias evolution from small random to [-0.263] is due to class imbalance")
