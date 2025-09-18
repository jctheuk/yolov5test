#!/usr/bin/env python3
"""
Minimal test for YOLOv5 classification
"""

import os
import sys
from pathlib import Path

# Add yolov5original to path
sys.path.append('yolov5original')

def minimal_test():
    """Minimal test"""
    
    print("🔍 Minimal test...")
    
    # Check if dataset exists
    dataset_path = Path("yolov5original/datasets/regurgitationV1-cls")
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    print("✅ Dataset found")
    
    # Check train directory
    train_path = dataset_path / "train"
    if train_path.exists():
        classes = ['A4C', 'PSAX', 'PLAX']
        for cls in classes:
            cls_path = train_path / cls
            if cls_path.exists():
                count = len(list(cls_path.glob("*.png")))
                print(f"  {cls}: {count} images")
    
    print("✅ Ready for training")

if __name__ == "__main__":
    minimal_test()
