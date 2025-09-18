#!/usr/bin/env python3
"""
Check the classification dataset structure
"""

import os
from pathlib import Path

def check_dataset():
    """Check the classification dataset structure"""
    
    dataset_path = "yolov5original/datasets/regurgitationV1-cls"
    classes = ['A4C', 'PSAX', 'PLAX']
    splits = ['train', 'val', 'test']
    
    print("🔍 Checking classification dataset structure...")
    print(f"Dataset path: {dataset_path}")
    
    total_images = 0
    
    for split in splits:
        print(f"\n📁 {split.upper()}:")
        split_path = Path(dataset_path) / split
        
        if not split_path.exists():
            print(f"❌ {split} directory not found")
            continue
            
        for cls in classes:
            cls_path = split_path / cls
            
            if not cls_path.exists():
                print(f"❌ {cls} directory not found in {split}")
                continue
                
            # Count images
            images = list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg"))
            count = len(images)
            total_images += count
            
            print(f"  {cls}: {count} images")
            
            # Check first few images
            if images:
                print(f"    Sample files: {[img.name for img in images[:3]]}")
    
    print(f"\n📊 Total images: {total_images}")
    
    # Check if dataset is valid for training
    if total_images == 0:
        print("❌ No images found in dataset!")
        return False
    
    # Check if we have images in train and val
    train_path = Path(dataset_path) / "train"
    val_path = Path(dataset_path) / "val"
    
    train_images = 0
    val_images = 0
    
    if train_path.exists():
        for cls in classes:
            cls_path = train_path / cls
            if cls_path.exists():
                train_images += len(list(cls_path.glob("*")))
    
    if val_path.exists():
        for cls in classes:
            cls_path = val_path / cls
            if cls_path.exists():
                val_images += len(list(cls_path.glob("*")))
    
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    
    if train_images == 0:
        print("❌ No training images found!")
        return False
        
    if val_images == 0:
        print("⚠️  No validation images found!")
    
    print("✅ Dataset structure looks good!")
    return True

if __name__ == "__main__":
    check_dataset()
