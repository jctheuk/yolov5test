#!/usr/bin/env python3
"""
Create separate Detection and Classification datasets from YOLOv5c joint datasets
for use with original YOLOv5

This script processes regurgitationV1-V5 datasets and creates:
1. Detection datasets: Keep only bounding box annotations (first line of labels)
2. Classification datasets: Create folder structure based on classification labels (second line)

Author: Generated for YOLOv5 original compatibility
"""

import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def create_detection_dataset(source_dir, target_dir):
    """
    Create detection dataset by keeping only detection annotations (first line)
    
    Args:
        source_dir (str): Source dataset directory (e.g., regurgitationV1)
        target_dir (str): Target detection dataset directory (e.g., regurgitationV1-Detection)
    """
    print(f"Creating detection dataset: {source_dir} -> {target_dir}")
    
    # Create target directory structure
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        source_split_path = Path(source_dir) / split
        if not source_split_path.exists():
            print(f"Warning: {source_split_path} does not exist, skipping...")
            continue
            
        target_split_path = target_path / split
        target_images_path = target_split_path / 'images'
        target_labels_path = target_split_path / 'labels'
        
        # Create directories
        target_images_path.mkdir(parents=True, exist_ok=True)
        target_labels_path.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        source_images_path = source_split_path / 'images'
        if source_images_path.exists():
            print(f"  Copying {split} images...")
            for img_file in tqdm(list(source_images_path.glob('*.png')), desc=f"{split} images"):
                shutil.copy2(img_file, target_images_path / img_file.name)
        
        # Process labels - keep only detection annotations (first line)
        source_labels_path = source_split_path / 'labels'
        if source_labels_path.exists():
            print(f"  Processing {split} labels...")
            for label_file in tqdm(list(source_labels_path.glob('*.txt')), desc=f"{split} labels"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Keep only the first line (detection annotation)
                    if lines:
                        detection_line = lines[0].strip()
                        if detection_line:  # Only write non-empty lines
                            target_label_file = target_labels_path / label_file.name
                            with open(target_label_file, 'w') as f:
                                f.write(detection_line + '\n')
                                
                except Exception as e:
                    print(f"Error processing {label_file}: {e}")
    
    # Create data.yaml for detection
    create_detection_yaml(target_dir)
    print(f"[SUCCESS] Detection dataset created: {target_dir}")

def create_classification_dataset(source_dir, target_dir):
    """
    Create classification dataset with folder structure based on classification labels
    
    Args:
        source_dir (str): Source dataset directory (e.g., regurgitationV1)  
        target_dir (str): Target classification dataset directory (e.g., regurgitationV1-Classification)
    """
    print(f"Creating classification dataset: {source_dir} -> {target_dir}")
    
    # Create target directory structure
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Classification class names
    class_names = ['A4C', 'PSAX', 'PLAX']
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        source_split_path = Path(source_dir) / split
        if not source_split_path.exists():
            print(f"Warning: {source_split_path} does not exist, skipping...")
            continue
            
        target_split_path = target_path / split
        
        # Create class directories
        class_dirs = {}
        for class_name in class_names:
            class_dir = target_split_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            class_dirs[class_name] = class_dir
        
        # Process images based on classification labels
        source_images_path = source_split_path / 'images'
        source_labels_path = source_split_path / 'labels'
        
        if source_images_path.exists() and source_labels_path.exists():
            print(f"  Processing {split} classification...")
            
            for img_file in tqdm(list(source_images_path.glob('*.png')), desc=f"{split} classification"):
                # Find corresponding label file
                label_file = source_labels_path / (img_file.stem + '.txt')
                
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Get classification annotation (second line)
                        if len(lines) >= 2:
                            classification_line = lines[1].strip()
                            class_values = list(map(int, classification_line.split()))
                            
                            # Find which class is active (value = 1)
                            if len(class_values) == 3:
                                active_class_idx = -1
                                for i, val in enumerate(class_values):
                                    if val == 1:
                                        active_class_idx = i
                                        break
                                
                                if active_class_idx >= 0:
                                    class_name = class_names[active_class_idx]
                                    target_img_path = class_dirs[class_name] / img_file.name
                                    shutil.copy2(img_file, target_img_path)
                                else:
                                    print(f"Warning: No active class found in {label_file}")
                            else:
                                print(f"Warning: Invalid classification format in {label_file}")
                        else:
                            print(f"Warning: No classification line found in {label_file}")
                            
                    except Exception as e:
                        print(f"Error processing {label_file}: {e}")
                else:
                    print(f"Warning: No label file found for {img_file}")
    
    print(f"[SUCCESS] Classification dataset created: {target_dir}")

def create_detection_yaml(dataset_dir):
    """Create data.yaml file for detection dataset"""
    yaml_content = f"""# YOLO Detection Dataset
# Detection: 4 classes (AR, MR, PR, TR) - Valve regurgitation types

# Dataset paths (relative to training script location)
train: {dataset_dir}/train/images
val: {dataset_dir}/valid/images  
test: {dataset_dir}/test/images

# Detection configuration
nc: 4  # number of detection classes
names: ['AR', 'MR', 'PR', 'TR']  # detection class names
"""
    
    yaml_path = Path(dataset_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"  Created: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Create Detection and Classification datasets from YOLOv5c joint datasets')
    parser.add_argument('--source-datasets', nargs='+', 
                       default=['regurgitationV1', 'regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5'],
                       help='Source dataset directories')
    parser.add_argument('--detection-only', action='store_true', help='Create only detection datasets')
    parser.add_argument('--classification-only', action='store_true', help='Create only classification datasets')
    
    args = parser.parse_args()
    
    # Process each source dataset
    for source_dataset in args.source_datasets:
        if not Path(source_dataset).exists():
            print(f"Warning: Source dataset {source_dataset} does not exist, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing: {source_dataset}")
        print(f"{'='*60}")
        
        # Create detection dataset
        if not args.classification_only:
            detection_dir = f"{source_dataset}-Detection"
            create_detection_dataset(source_dataset, detection_dir)
        
        # Create classification dataset  
        if not args.detection_only:
            classification_dir = f"{source_dataset}-Classification"
            create_classification_dataset(source_dataset, classification_dir)
    
    print(f"\n{'='*60}")
    print("[SUCCESS] All datasets processed successfully!")
    print(f"{'='*60}")
    
    print("\nCreated datasets:")
    for source_dataset in args.source_datasets:
        if Path(source_dataset).exists():
            if not args.classification_only:
                print(f"  [DETECTION] {source_dataset}-Detection/")
            if not args.detection_only:
                print(f"  [CLASSIFICATION] {source_dataset}-Classification/")

if __name__ == '__main__':
    main()
