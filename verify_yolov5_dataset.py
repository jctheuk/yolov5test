#!/usr/bin/env python3
"""
Verify the YOLOv5 dataset structure and format
"""

import os
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify the YOLOv5 dataset"""
    
    print(f"=== Verifying YOLOv5 Dataset: {dataset_path} ===\n")
    
    # Check directory structure
    for split in ['train', 'valid', 'test']:
        images_dir = Path(dataset_path) / split / 'images'
        labels_dir = Path(dataset_path) / split / 'labels'
        
        if not images_dir.exists():
            print(f"❌ {split}/images directory not found")
            continue
            
        if not labels_dir.exists():
            print(f"❌ {split}/labels directory not found")
            continue
        
        # Count files
        image_files = list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"📁 {split.upper()} split:")
        print(f"   Images: {len(image_files)}")
        print(f"   Labels: {len(label_files)}")
        
        if len(image_files) != len(label_files):
            print(f"   ⚠️  Mismatch: {len(image_files)} images vs {len(label_files)} labels")
        else:
            print(f"   ✅ Perfect match")
        
        # Check label format
        if label_files:
            sample_label = label_files[0]
            with open(sample_label, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n')
                
                print(f"   📝 Sample label format: {len(lines)} line(s)")
                if len(lines) == 1:
                    parts = lines[0].split()
                    if len(parts) == 5:
                        print(f"   ✅ Correct YOLOv5 format: class_id x_center y_center width height")
                    else:
                        print(f"   ❌ Wrong format: expected 5 values, got {len(parts)}")
                else:
                    print(f"   ❌ Wrong format: expected 1 line, got {len(lines)}")
        
        print()
    
    # Check data.yaml
    yaml_file = Path(dataset_path) / 'data.yaml'
    if yaml_file.exists():
        print(f"📄 data.yaml: ✅ Found")
        with open(yaml_file, 'r') as f:
            content = f.read()
            print(f"   Content preview:")
            for line in content.split('\n')[:10]:
                if line.strip():
                    print(f"   {line}")
    else:
        print(f"📄 data.yaml: ❌ Not found")

if __name__ == "__main__":
    verify_dataset("regurgitation-yolov5") 