#!/usr/bin/env python3
"""
Safe training script that uses yolov5s-cls.pt weights with proper handling
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add yolov5c to path
sys.path.append('/work/jonchang3909/yolov5test/yolov5c')

def safe_load_cls_weights(model, cls_weights_path, device):
    """
    Safely load yolov5s-cls.pt weights into detection+classification model
    Only loads compatible backbone weights, skips incompatible layers
    """
    print(f"🔄 Loading weights from {cls_weights_path}")
    
    # Load classification weights
    cls_ckpt = torch.load(cls_weights_path, map_location='cpu')
    cls_state_dict = cls_ckpt['model'].float().state_dict()
    
    # Get target model state dict
    target_state_dict = model.state_dict()
    
    # Find compatible weights (mainly backbone layers)
    compatible_weights = {}
    incompatible_count = 0
    
    for key, value in cls_state_dict.items():
        if key in target_state_dict:
            # Check if shapes match
            if value.shape == target_state_dict[key].shape:
                compatible_weights[key] = value
            else:
                incompatible_count += 1
        else:
            incompatible_count += 1
    
    print(f"📊 Weight Transfer Summary:")
    print(f"   Compatible weights: {len(compatible_weights)}")
    print(f"   Incompatible keys: {incompatible_count}")
    
    # Load compatible weights with strict=False to ignore incompatible ones
    missing_keys, unexpected_keys = model.load_state_dict(compatible_weights, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys (will use random initialization): {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)}")
    
    print(f"✅ Successfully loaded {len(compatible_weights)} compatible weights")
    return model

def create_safe_training_command():
    """Create a safe training command that uses yolov5s-cls.pt"""
    
    command = '''#!/bin/bash

# Safe training with yolov5s-cls.pt weights
cd /work/jonchang3909/yolov5test/yolov5c/

echo "🚀 Starting training with yolov5s-cls.pt weights..."

python train_classification_task.py \\
    --data ../regurgitationV1/data.yaml \\
    --weights yolov5s-cls.pt \\
    --cfg models/yolov5sc.yaml \\
    --epochs 300 \\
    --batch-size 64 \\
    --imgsz 416 \\
    --name classifyloss_cls_safe \\
    --cache \\
    --nosave \\
    --lr0 0.001 \\
    --momentum 0.937 \\
    --weight-decay 0.0005 \\
    --warmup-epochs 3.0 \\
    --patience 0

echo "✅ Training completed!"
'''
    
    with open('train_cls_safe.sh', 'w') as f:
        f.write(command)
    
    print("📝 Created train_cls_safe.sh")
    print("💡 Run: chmod +x train_cls_safe.sh && ./train_cls_safe.sh")

if __name__ == '__main__':
    create_safe_training_command()

