#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert YOLOv5 Segmentation Format to Detection Format
Converts datasets 2-5 from segmentation (polygons) to detection (bounding boxes)
While preserving classification one-hot encoding
"""

import os
import shutil
import numpy as np
from pathlib import Path
import argparse


def polygon_to_bbox(polygon_points):
    """
    Convert polygon coordinates to bounding box format
    Args:
        polygon_points: List of alternating x,y coordinates [x1,y1,x2,y2,...]
    Returns:
        tuple: (x_center, y_center, width, height) in normalized coordinates
    """
    if len(polygon_points) < 6:  # Need at least 3 points (6 coordinates)
        raise ValueError(f"Invalid polygon: need at least 3 points, got {len(polygon_points)//2}")
    
    # Extract x and y coordinates
    x_coords = [float(polygon_points[i]) for i in range(0, len(polygon_points), 2)]
    y_coords = [float(polygon_points[i]) for i in range(1, len(polygon_points), 2)]
    
    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Convert to YOLO format (center coordinates + width/height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_label_file(input_path, output_path):
    """
    Convert a single label file from segmentation to detection format
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Empty line
            converted_lines.append('')
            continue
            
        parts = line.split()
        
        # Check if this is a classification line (one-hot encoding)
        if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
            # This is the classification line, keep as is
            converted_lines.append(line)
            continue
            
        # This should be a segmentation line
        if len(parts) < 7:  # class + at least 3 points (6 coordinates)
            print(f"Warning: Skipping invalid segmentation line in {input_path}: {line}")
            continue
            
        try:
            class_id = int(parts[0])
            polygon_coords = [float(p) for p in parts[1:]]
            
            # Convert polygon to bounding box
            x_center, y_center, width, height = polygon_to_bbox(polygon_coords)
            
            # Format as detection line
            detection_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            converted_lines.append(detection_line)
            
        except (ValueError, IndexError) as e:
            print(f"Error processing line in {input_path}: {line}")
            print(f"Error: {e}")
            continue
    
    # Write converted file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')


def create_data_yaml(dataset_name, output_dir):
    """
    Create data.yaml file for the converted dataset
    """
    yaml_content = f"""# YOLO Multi-Task Dataset (Detection + Classification)
# Detection: 4 classes (AR, MR, PR, TR) - Valve regurgitation types  
# Classification: 3 classes (PSAX, PLAX, A4C) - Echocardiogram views

# Dataset paths (relative to yolov5c directory where training script runs)
train: ../{dataset_name}/train/images
val: ../{dataset_name}/valid/images  
test: ../{dataset_name}/test/images

# Detection configuration
nc: 4  # number of detection classes
names: ['AR', 'MR', 'PR', 'TR']  # detection class names

# Classification configuration
num_cls: 3  # number of classification classes
cls_names: ['A4C', 'PSAX', 'PLAX']  # classification class names
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Created data.yaml: {yaml_path}")


def convert_dataset(input_dir, output_dir, dataset_name):
    """
    Convert entire dataset from segmentation to detection format
    """
    print(f"Converting {input_dir} -> {output_dir}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images and convert labels for each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        if not os.path.exists(input_split_dir):
            print(f"Warning: {input_split_dir} not found, skipping...")
            continue
            
        # Create output directories
        os.makedirs(os.path.join(output_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_split_dir, 'labels'), exist_ok=True)
        
        # Copy images
        input_images_dir = os.path.join(input_split_dir, 'images')
        output_images_dir = os.path.join(output_split_dir, 'images')
        
        if os.path.exists(input_images_dir):
            for img_file in os.listdir(input_images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(
                        os.path.join(input_images_dir, img_file),
                        os.path.join(output_images_dir, img_file)
                    )
        
        # Convert labels
        input_labels_dir = os.path.join(input_split_dir, 'labels')
        output_labels_dir = os.path.join(output_split_dir, 'labels')
        
        if os.path.exists(input_labels_dir):
            label_count = 0
            for label_file in os.listdir(input_labels_dir):
                if label_file.endswith('.txt'):
                    input_label_path = os.path.join(input_labels_dir, label_file)
                    output_label_path = os.path.join(output_labels_dir, label_file)
                    
                    convert_label_file(input_label_path, output_label_path)
                    label_count += 1
            
            print(f"  {split}: Converted {label_count} label files")
    
    # Create data.yaml
    create_data_yaml(dataset_name, output_dir)
    print(f"Dataset conversion completed: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation datasets to detection format')
    parser.add_argument('--input-base', type=str, default='.',
                        help='Base directory containing input datasets')
    parser.add_argument('--datasets', nargs='+', default=['2', '3', '4', '5'],
                        help='Dataset numbers to convert (default: 2 3 4 5)')
    args = parser.parse_args()
    
    base_dir = args.input_base
    
    # Define input and output mappings
    dataset_mappings = {
        '2': ('Regurgitation 2025_Regurgitation-YOLODataset-2/Regurgitation-YOLODataset-2', 'regurgitationV2'),
        '3': ('Regurgitation 2025_Regurgitation-YOLODataset-3/Regurgitation-YOLODataset-3', 'regurgitationV3'),
        '4': ('Regurgitation 2025_Regurgitation-YOLODataset-4/Regurgitation-YOLODataset-4', 'regurgitationV4'),
        '5': ('Regurgitation 2025_Regurgitation-YOLODataset-5/Regurgitation-YOLODataset-5', 'regurgitationV5'),
    }
    
    print("=== YOLO Segmentation to Detection Converter ===")
    print("Converting datasets from polygon segmentation to bounding box detection")
    print("Preserving classification one-hot encoding\n")
    
    for dataset_num in args.datasets:
        if dataset_num not in dataset_mappings:
            print(f"Error: Unknown dataset number '{dataset_num}'. Available: {list(dataset_mappings.keys())}")
            continue
            
        input_subdir, output_name = dataset_mappings[dataset_num]
        input_dir = os.path.join(base_dir, input_subdir)
        output_dir = os.path.join(base_dir, output_name)
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            continue
            
        try:
            convert_dataset(input_dir, output_dir, output_name)
            print(f"[SUCCESS] Successfully converted dataset {dataset_num} -> {output_name}\n")
            
        except Exception as e:
            print(f"[ERROR] Error converting dataset {dataset_num}: {e}\n")
    
    print("=== Conversion Summary ===")
    print("Converted datasets can be used with YOLOv5WithClassification training")
    print("Each dataset includes:")
    print("- Detection bounding boxes (converted from segmentation polygons)")
    print("- Classification one-hot encoding (preserved from original)")
    print("- Complete train/valid/test splits")
    print("- data.yaml configuration file")


if __name__ == "__main__":
    main()
