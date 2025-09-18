#!/usr/bin/env python3
"""
Create a simple classification dataset for testing
"""

import os
import shutil
from pathlib import Path

def create_test_classification_dataset():
    """Create a small test classification dataset"""
    
    # Source dataset
    source_dataset = "Regurgitation-YOLODataset-Detection"
    
    # Output dataset
    output_dataset = "yolov5original/datasets/regurgitation-test"
    
    # Classes
    classes = ['A4C', 'PSAX', 'PLAX']
    
    print("🔄 Creating test classification dataset...")
    
    # Create output directories
    for split in ['train', 'val']:
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and val splits (skip test for now)
    for split in ['train', 'val']:
        print(f"\n📁 Processing {split} split...")
        
        source_images = Path(source_dataset) / split / "images"
        source_labels = Path(source_dataset) / split / "labels"
        
        if not source_images.exists():
            print(f"❌ Source images directory not found: {source_images}")
            continue
            
        # Get first 10 images for testing
        image_files = list(source_images.glob("*.png"))[:10]
        print(f"Processing {len(image_files)} images for testing")
        
        for img_path in image_files:
            label_path = source_labels / (img_path.stem + ".txt")
            
            if not label_path.exists():
                continue
                
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        cls_label = lines[1].strip().split()
                        if len(cls_label) == 3:
                            cls_idx = cls_label.index('1')
                            cls_name = classes[cls_idx]
                            
                            # Copy image
                            dest_path = Path(output_dataset) / split / cls_name / img_path.name
                            shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"❌ Error processing {label_path}: {e}")
    
    # Count files
    print(f"\n📊 Test dataset statistics:")
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            print(f"  {cls}: {count} images")
    
    print(f"\n✅ Test dataset created at: {output_dataset}")
    return output_dataset

if __name__ == "__main__":
    create_test_classification_dataset()
