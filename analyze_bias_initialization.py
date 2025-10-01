"""
Analyze where the bias gets its value in the actual training flow

Your config shows:
- cfg: models/yolov5sc_classify_backbone.yaml (has classification head)
- weights: yolov5s.pt (no classification head)

The question: What's the classification head bias after this initialization?
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
sys.path.append('yolov5c')

from models.yolo import Model
from utils.general import intersect_dicts

def test_actual_flow():
    """Test the exact flow from train_classification_task.py"""
    print("=" * 60)
    print("TESTING ACTUAL INITIALIZATION FLOW")
    print("=" * 60)
    
    # Your actual configuration
    cfg = 'yolov5c/models/yolov5sc_classify_backbone.yaml'
    weights = 'yolov5s.pt'
    
    # Load data.yaml
    with open('regurgitationV1/data.yaml', 'r') as f:
        data_dict = yaml.safe_load(f)
    nc = data_dict['nc']
    
    print(f"\nConfiguration:")
    print(f"  cfg: {cfg}")
    print(f"  weights: {weights}")
    print(f"  nc (detection classes): {nc}")
    
    # Step 1: Load checkpoint
    print(f"\nStep 1: Loading checkpoint...")
    ckpt = torch.load(weights, map_location='cpu')
    
    # Step 2: Create model (line 693)
    print(f"\nStep 2: Creating model from cfg...")
    model = Model(cfg, ch=3, nc=nc, anchors=None)
    
    # Find classification head
    classification_head = None
    classification_head_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == 3:
            classification_head = module
            classification_head_name = name
            break
    
    if classification_head is None:
        print(f"  ERROR: No classification head found!")
        return
    
    print(f"\nClassification head found: {classification_head_name}")
    print(f"  After model creation:")
    print(f"    Bias (PyTorch default): {classification_head.bias.data}")
    initial_bias = classification_head.bias.data.clone()
    
    # Step 3: Load weights from checkpoint (lines 735-737)
    print(f"\nStep 3: Loading weights from {weights}...")
    
    # Get state dict
    csd = ckpt['model'].float().state_dict()
    
    # Intersect with model state dict
    exclude = ['anchor']
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    
    # Check what's in csd for classification head
    classification_keys = [k for k in csd.keys() if classification_head_name in k]
    print(f"  Classification head keys in checkpoint: {classification_keys}")
    
    if not classification_keys:
        print(f"  ✅ No classification head in yolov5s.pt")
        print(f"  Classification head will keep its PyTorch default bias")
    
    # Load state dict
    print(f"\nStep 4: Loading state dict (strict=False)...")
    missing, unexpected = model.load_state_dict(csd, strict=False)
    
    print(f"  Missing keys: {len(missing)} (includes classification head)")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    # Check bias after loading
    print(f"\n  After load_state_dict:")
    print(f"    Bias: {classification_head.bias.data}")
    
    # Compare
    if torch.allclose(classification_head.bias.data, initial_bias, atol=1e-6):
        print(f"  ✅ Bias UNCHANGED (PyTorch default preserved)")
        print(f"\n  CONCLUSION: Bias starts as small random values, NOT [0,0,0]")
    else:
        print(f"  WARNING: Bias changed during weight loading!")
        print(f"    Before: {initial_bias}")
        print(f"    After: {classification_head.bias.data}")

def check_if_checkpoint_has_classification():
    """Check what's actually in your trained checkpoint"""
    print("\n" + "=" * 60)
    print("CHECKING YOUR TRAINED CHECKPOINT")
    print("=" * 60)
    
    checkpoint_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {ckpt.keys()}")
    
    if 'model' in ckpt:
        # Check if it's a state_dict or model object
        if hasattr(ckpt['model'], 'state_dict'):
            state_dict = ckpt['model'].state_dict()
        else:
            state_dict = ckpt['model']
        
        # Find classification head bias in checkpoint
        for key in state_dict.keys():
            if 'linear' in key.lower() and 'bias' in key.lower():
                print(f"\nFound in checkpoint: {key}")
                print(f"  Value: {state_dict[key]}")
                
                if key.endswith('bias') and state_dict[key].shape[0] == 3:
                    print(f"\n  This is the classification head bias!")
                    print(f"  Values: {state_dict[key]}")
                    
                    if torch.allclose(state_dict[key], torch.zeros(3), atol=1e-4):
                        print(f"  ⚠️  Bias is ZERO or near-zero in the checkpoint!")
                        print(f"     This suggests it was initialized to zero")
                    else:
                        print(f"  Bias has non-zero values (expected after training)")

if __name__ == "__main__":
    test_actual_flow()
    check_if_checkpoint_has_classification()
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print("\nThe classification head bias initialization flow:")
    print("  1. Model created from yolov5sc_classify_backbone.yaml")
    print("  2. YOLOv5WithClassification uses PyTorch default → small random bias")
    print("  3. Load weights from yolov5s.pt → no classification weights, bias unchanged")
    print("  4. Bias should be SMALL RANDOM, not [0, 0, 0]")
    print("\nIf you see [0, 0, 0], check:")
    print("  - Are you resuming from a checkpoint with zero bias?")
    print("  - Is there code that explicitly resets bias?")
    print("\nIf bias starts small random and becomes [-0.263], that's due to")
    print("class imbalance during training - use class weights to fix!")