#!/usr/bin/env python3
"""
Create classification dataset for YOLOv5 original classification module
Converts detection dataset to folder-per-class format for classification training
"""

import os
import shutil
from pathlib import Path
import random

def create_classification_dataset():
    """Create classification dataset in folder-per-class format"""
    
    # Source dataset (detection format)
    source_dataset = "Regurgitation-YOLODataset-Detection"
    
    # Output dataset (classification format)
    output_dataset = "yolov5original/datasets/regurgitation-classification"
    
    # Classification classes
    classes = ['A4C', 'PSAX', 'PLAX']
    
    print("🔄 Creating classification dataset...")
    print(f"Source: {source_dataset}")
    print(f"Output: {output_dataset}")
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Processing {split} split...")
        
        source_images = Path(source_dataset) / split / "images"
        source_labels = Path(source_dataset) / split / "labels"
        
        if not source_images.exists():
            print(f"❌ Source images directory not found: {source_images}")
            continue
            
        # Get all image files
        image_files = list(source_images.glob("*.png")) + list(source_images.glob("*.jpg"))
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in image_files:
            # Get corresponding label file
            label_path = source_labels / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"⚠️  No label file for {img_path.name}")
                continue
            
            # Read classification label (second line of label file)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        # Parse classification label (one-hot encoding)
                        cls_label = lines[1].strip().split()
                        if len(cls_label) == 3:
                            # Convert one-hot to class index
                            cls_idx = cls_label.index('1')
                            cls_name = classes[cls_idx]
                            
                            # Copy image to appropriate class folder
                            dest_path = Path(output_dataset) / split / cls_name / img_path.name
                            shutil.copy2(img_path, dest_path)
                        else:
                            print(f"⚠️  Invalid classification label in {label_path}")
                    else:
                        print(f"⚠️  No classification label in {label_path}")
            except Exception as e:
                print(f"❌ Error processing {label_path}: {e}")
    
    # Count files in each class
    print(f"\n📊 Dataset statistics:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            print(f"  {cls}: {count} images")
    
    print(f"\n✅ Classification dataset created at: {output_dataset}")
    return output_dataset

if __name__ == "__main__":
    create_classification_dataset()
