#!/usr/bin/env python3
"""
Quick test to check dataset and run 1 epoch training
"""

import os
import sys
from pathlib import Path

# Add yolov5original to path
sys.path.append('yolov5original')

def quick_test():
    """Quick test of the classification dataset and training"""
    
    print("🔍 Quick test of YOLOv5 classification...")
    
    # Check dataset structure
    dataset_path = "yolov5original/datasets/regurgitationV1-cls"
    classes = ['A4C', 'PSAX', 'PLAX']
    
    print(f"Checking dataset at: {dataset_path}")
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_path = Path(dataset_path) / split / cls
            if cls_path.exists():
                count = len(list(cls_path.glob("*.png")))
                print(f"  {cls}: {count} images")
            else:
                print(f"  {cls}: Directory not found")
    
    # Try to run training
    print(f"\n🚀 Starting 1-epoch training test...")
    
    os.chdir('yolov5original')
    
    # Import and run training
    from classify.train import main
    import argparse
    
    # Create arguments
    args = argparse.Namespace(
        data='datasets/regurgitationV1-cls',
        model='yolov5s-cls.pt',
        epochs=1,
        batch_size=4,
        imgsz=416,
        device='cpu',
        workers=0,
        name='test_1epoch',
        project='runs/train-cls',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        decay=5e-05,
        label_smoothing=0.1,
        cutoff=None,
        dropout=None,
        verbose=False,
        seed=0,
        local_rank=-1,
        nosave=False,
        cache=None
    )
    
    try:
        main(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
