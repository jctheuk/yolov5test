#!/usr/bin/env python3
"""
Clean corrupted images from the classification dataset
"""

import cv2
import os
from pathlib import Path

def clean_dataset():
    """Remove corrupted images from the dataset"""
    
    dataset_path = Path("yolov5original/datasets/regurgitationV1-cls")
    classes = ['A4C', 'PSAX', 'PLAX']
    splits = ['train', 'val', 'test']
    
    print("🧹 Cleaning corrupted images from dataset...")
    print(f"Dataset path: {dataset_path}")
    
    total_removed = 0
    total_checked = 0
    
    for split in splits:
        print(f"\n📁 Checking {split} split...")
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"❌ {split} directory not found")
            continue
            
        for cls in classes:
            cls_path = split_path / cls
            
            if not cls_path.exists():
                print(f"❌ {cls} directory not found in {split}")
                continue
                
            # Check all images in this class
            images = list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg"))
            print(f"  {cls}: Checking {len(images)} images...")
            
            removed_count = 0
            for img_file in images:
                total_checked += 1
                try:
                    # Try to read the image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        print(f"    ❌ Removing corrupted: {img_file.name}")
                        img_file.unlink()
                        removed_count += 1
                        total_removed += 1
                    else:
                        # Additional check - try to decode
                        success, _ = cv2.imencode('.png', img)
                        if not success:
                            print(f"    ❌ Removing corrupted: {img_file.name}")
                            img_file.unlink()
                            removed_count += 1
                            total_removed += 1
                except Exception as e:
                    print(f"    ❌ Removing corrupted: {img_file.name} (Error: {e})")
                    try:
                        img_file.unlink()
                        removed_count += 1
                        total_removed += 1
                    except:
                        pass
            
            if removed_count > 0:
                print(f"    🗑️  Removed {removed_count} corrupted images from {cls}")
            else:
                print(f"    ✅ All images in {cls} are valid")
    
    print(f"\n📊 Cleaning Summary:")
    print(f"  Total images checked: {total_checked}")
    print(f"  Total images removed: {total_removed}")
    print(f"  Valid images remaining: {total_checked - total_removed}")
    
    if total_removed > 0:
        print(f"\n✅ Dataset cleaned! Removed {total_removed} corrupted images.")
        print("You can now re-run the training.")
    else:
        print(f"\n✅ Dataset is clean! No corrupted images found.")
    
    return total_removed

if __name__ == "__main__":
    clean_dataset()

