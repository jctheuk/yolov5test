#!/usr/bin/env python3
"""
Clone Regurgitation-YOLODataset-Detection and remove one-hot encoding to create standard YOLOv5 dataset
"""

import os
import shutil
from pathlib import Path

def remove_classification_line(input_path, output_path):
    """
    Remove the classification line (second line) from label files
    Keep only the detection line (first line)
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 1:
        print(f"Warning: {input_path} is empty")
        return False
    
    # Keep only the first line (detection format)
    detection_line = lines[0].strip()
    if not detection_line:
        print(f"Warning: {input_path} has empty detection line")
        return False
    
    # Write only the detection line
    with open(output_path, 'w') as f:
        f.write(detection_line + "\n")
    
    return True

def clone_dataset(input_dir, output_dir):
    """
    Clone dataset and remove classification lines from labels
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        input_labels_dir = input_path / split / 'labels'
        input_images_dir = input_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        output_images_dir = output_path / split / 'images'
        
        if not input_labels_dir.exists():
            print(f"Warning: {input_labels_dir} does not exist, skipping...")
            continue
        
        print(f"Processing {split} split...")
        
        # Process label files (remove classification lines)
        label_files = list(input_labels_dir.glob('*.txt'))
        processed_count = 0
        
        for label_file in label_files:
            output_label_file = output_labels_dir / label_file.name
            
            try:
                success = remove_classification_line(label_file, output_label_file)
                if success:
                    processed_count += 1
                else:
                    print(f"Failed to process {label_file}")
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"Processed {processed_count} label files for {split} split")
        
        # Copy image files
        if input_images_dir.exists():
            image_files = list(input_images_dir.glob('*.*'))
            copied_count = 0
            
            for image_file in image_files:
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    output_image_file = output_images_dir / image_file.name
                    try:
                        shutil.copy2(image_file, output_image_file)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {image_file}: {e}")
            
            print(f"Copied {copied_count} image files for {split} split")
    
    # Create data.yaml file
    create_data_yaml(output_path, input_path)

def create_data_yaml(output_dir, input_dir):
    """
    Create data.yaml file for the YOLOv5 dataset
    """
    # Read original data.yaml to get class names
    input_yaml = input_dir / 'data.yaml'
    class_names = ['AR', 'MR', 'PR', 'TR']  # Default names
    
    if input_yaml.exists():
        import yaml
        with open(input_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                class_names = data['names']
    
    # Create new data.yaml for YOLOv5
    output_yaml = output_dir / 'data.yaml'
    yaml_content = f"""# YOLOv5 Dataset Configuration
names:
{chr(10).join([f"- {name}" for name in class_names])}
nc: {len(class_names)}
train: {output_dir}/train/images
val: {output_dir}/valid/images
test: {output_dir}/test/images
"""
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {output_yaml}")

def main():
    """Main function"""
    # Define paths
    input_dataset = "Regurgitation-YOLODataset-Detection"
    output_dataset = "regurgitation-yolov5"
    
    print(f"Creating YOLOv5 dataset from {input_dataset}")
    print("This will remove classification lines and keep only detection format")
    
    # Check if input dataset exists
    if not Path(input_dataset).exists():
        print(f"Error: Input dataset {input_dataset} not found!")
        return
    
    # Check if output directory already exists
    if Path(output_dataset).exists():
        print(f"Warning: Output directory {output_dataset} already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        shutil.rmtree(output_dataset)
    
    # Clone and process dataset
    clone_dataset(input_dataset, output_dataset)
    
    print(f"\nYOLOv5 dataset created successfully!")
    print(f"Output dataset: {output_dataset}")
    print(f"Format: Standard YOLOv5 detection (class_id x_center y_center width height)")
    print(f"Classification lines have been removed.")

if __name__ == "__main__":
    main() 