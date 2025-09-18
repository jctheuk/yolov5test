#!/usr/bin/env python3
"""
Create YOLOv5 classification dataset from detection dataset
Converts detection format to folder-per-class format for YOLOv5 classify module
"""

import os
import shutil
from pathlib import Path
import random

def create_yolov5_classification_dataset():
    """Create YOLOv5 classification dataset in folder-per-class format"""
    
    # Source dataset (detection format)
    source_dataset = "regurgitationV1"
    
    # Output dataset (classification format)
    output_dataset = "yolov5original/datasets/regurgitationV1-cls"
    
    # Classification classes (from data.yaml)
    classes = ['A4C', 'PSAX', 'PLAX']
    
    print("🔄 Creating YOLOv5 Classification Dataset...")
    print(f"Source: {source_dataset}")
    print(f"Output: {output_dataset}")
    print(f"Classes: {classes}")
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {cls_dir}")
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n📁 Processing {split} split...")
        
        source_images = Path(source_dataset) / split / "images"
        source_labels = Path(source_dataset) / split / "labels"
        
        if not source_images.exists():
            print(f"❌ Source images directory not found: {source_images}")
            continue
            
        # Get all image files
        image_files = list(source_images.glob("*.png")) + list(source_images.glob("*.jpg"))
        print(f"Found {len(image_files)} images")
        
        processed_count = 0
        error_count = 0
        
        # Process each image
        for img_path in image_files:
            # Get corresponding label file
            label_path = source_labels / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"⚠️  No label file for {img_path.name}")
                error_count += 1
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
                            if '1' in cls_label:
                                cls_idx = cls_label.index('1')
                                cls_name = classes[cls_idx]
                                
                                # Copy image to appropriate class folder
                                dest_path = Path(output_dataset) / split / cls_name / img_path.name
                                shutil.copy2(img_path, dest_path)
                                processed_count += 1
                                
                                if processed_count % 100 == 0:
                                    print(f"  Processed {processed_count} images...")
                            else:
                                print(f"⚠️  No '1' found in classification label: {cls_label}")
                                error_count += 1
                        else:
                            print(f"⚠️  Invalid classification label format in {label_path}: {cls_label}")
                            error_count += 1
                    else:
                        print(f"⚠️  No classification label in {label_path}")
                        error_count += 1
            except Exception as e:
                print(f"❌ Error processing {label_path}: {e}")
                error_count += 1
        
        print(f"✅ {split}: Processed {processed_count} images, {error_count} errors")
    
    # Count files in each class
    print(f"\n📊 Final Dataset Statistics:")
    total_images = 0
    for split in ['train', 'valid', 'test']:
        print(f"\n{split.upper()}:")
        split_total = 0
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            print(f"  {cls}: {count} images")
            split_total += count
        print(f"  Total: {split_total} images")
        total_images += split_total
    
    print(f"\n🎯 Total dataset: {total_images} images")
    print(f"✅ YOLOv5 classification dataset created at: {output_dataset}")
    
    # Verify the structure
    print(f"\n🔍 Verifying dataset structure...")
    for split in ['train', 'valid', 'test']:
        for cls in classes:
            cls_dir = Path(output_dataset) / split / cls
            if cls_dir.exists():
                files = list(cls_dir.glob("*"))
                if files:
                    print(f"✅ {split}/{cls}: {len(files)} images")
                else:
                    print(f"⚠️  {split}/{cls}: No images found")
            else:
                print(f"❌ {split}/{cls}: Directory not found")
    
    return output_dataset

if __name__ == "__main__":
    create_yolov5_classification_dataset()
