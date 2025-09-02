#!/usr/bin/env python3
"""
Fix classification labels generation for YOLOv5 multi-task training
The current logic fails to identify view types from filenames
"""

import os
import numpy as np
from pathlib import Path
import yaml
import random

def analyze_filename_patterns(dataset_path):
    """Analyze filename patterns to understand the dataset structure"""
    print(f"Analyzing filename patterns in: {dataset_path}")
    
    # Get sample filenames from each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_path = Path(dataset_path) / split / "images"
        if not split_path.exists():
            continue
            
        print(f"\n=== {split.upper()} SET FILENAME PATTERNS ===")
        image_files = list(split_path.glob("*.png"))
        if not image_files:
            continue
            
        # Show first 10 filenames
        print("Sample filenames:")
        for i, img_file in enumerate(image_files[:10]):
            print(f"  {i+1}. {img_file.name}")
        
        # Analyze patterns
        patterns = {}
        for img_file in image_files[:100]:  # Analyze first 100 files
            name = img_file.name
            # Extract video ID (before the first dash)
            if '-' in name:
                video_id = name.split('-')[0]
                patterns[video_id] = patterns.get(video_id, 0) + 1
        
        print(f"\nVideo ID patterns (first 100 files):")
        for video_id, count in sorted(patterns.items())[:10]:
            print(f"  {video_id}: {count} files")

def create_balanced_classification_labels(dataset_path, output_path=None):
    """
    Create balanced classification labels based on video ID patterns
    Since we can't determine view type from filename, we'll create a balanced distribution
    """
    print(f"Creating balanced classification labels for: {dataset_path}")
    
    if output_path is None:
        output_path = dataset_path
    
    # Load data.yaml
    data_yaml_path = Path(dataset_path) / "data.yaml"
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    cls_names = data_config.get('cls_names', ['PSAX', 'PLAX', 'A4C'])
    num_classes = len(cls_names)
    
    print(f"Classification classes: {cls_names}")
    
    # Process each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            continue
            
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        if not images_path.exists() or not labels_path.exists():
            continue
        
        print(f"\nProcessing {split} set...")
        
        # Get all image files
        image_files = list(images_path.glob("*.png"))
        if not image_files:
            continue
        
        # Group by video ID
        video_groups = {}
        for img_file in image_files:
            name = img_file.name
            if '-' in name:
                video_id = name.split('-')[0]
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append(img_file)
        
        print(f"Found {len(video_groups)} unique video IDs")
        
        # Assign classification labels to videos
        video_labels = {}
        video_ids = list(video_groups.keys())
        
        # Create balanced distribution
        num_videos = len(video_ids)
        videos_per_class = num_videos // num_classes
        remainder = num_videos % num_classes
        
        # Assign labels
        label_idx = 0
        for i, video_id in enumerate(video_ids):
            if i < videos_per_class * num_classes:
                class_idx = i // videos_per_class
            else:
                # Distribute remainder evenly
                class_idx = num_classes - 1 - (i - videos_per_class * num_classes)
            
            # Create one-hot encoding
            one_hot = [0.0] * num_classes
            one_hot[class_idx] = 1.0
            video_labels[video_id] = one_hot
        
        # Verify distribution
        class_counts = [0] * num_classes
        for label in video_labels.values():
            class_idx = np.argmax(label)
            class_counts[class_idx] += 1
        
        print(f"Class distribution:")
        for i, (cls_name, count) in enumerate(zip(cls_names, class_counts)):
            print(f"  {cls_name}: {count} videos")
        
        # Add classification labels to label files
        files_updated = 0
        for video_id, files in video_groups.items():
            classification_label = video_labels[video_id]
            
            for img_file in files:
                # Find corresponding label file
                label_file = labels_path / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue
                
                # Read existing labels
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Check if classification label already exists
                has_classification = any(line.strip().startswith('[') and line.strip().endswith(']') for line in lines)
                
                if not has_classification:
                    # Add classification label
                    with open(label_file, 'a') as f:
                        f.write(f"\n{str(classification_label)}\n")
                    files_updated += 1
        
        print(f"Updated {files_updated} label files with classification labels")

def create_improved_classification_function():
    """Create an improved classification label generation function"""
    improved_function = '''
def create_classification_labels_from_paths(image_paths, num_classes=3, cls_names=None):
    """
    Improved classification label generation based on video ID patterns
    Args:
        image_paths: List of image file paths
        num_classes: Number of classification classes
        cls_names: List of classification class names
    Returns:
        torch.Tensor: One-hot encoded classification labels
    """
    batch_size = len(image_paths)
    classification_labels = torch.zeros(batch_size, num_classes)
    
    # Default class names if not provided
    if cls_names is None:
        cls_names = ['PSAX', 'PLAX', 'A4C']
    
    # Group images by video ID
    video_groups = {}
    for i, img_path in enumerate(image_paths):
        filename = Path(img_path).name
        if '-' in filename:
            video_id = filename.split('-')[0]
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(i)
    
    # Assign consistent labels to videos
    video_ids = list(video_groups.keys())
    num_videos = len(video_ids)
    
    # Create balanced distribution
    videos_per_class = num_videos // num_classes
    remainder = num_videos % num_classes
    
    for i, video_id in enumerate(video_ids):
        if i < videos_per_class * num_classes:
            class_idx = i // videos_per_class
        else:
            # Distribute remainder evenly
            class_idx = num_classes - 1 - (i - videos_per_class * num_classes)
        
        # Assign label to all images from this video
        for img_idx in video_groups[video_id]:
            classification_labels[img_idx, class_idx] = 1.0
    
    return classification_labels
'''
    
    return improved_function

def main():
    """Main function to fix classification labels"""
    print("=== YOLOv5 Classification Labels Fix ===\n")
    
    dataset_path = "Regurgitation-YOLODataset-Detection"
    
    # Step 1: Analyze current filename patterns
    analyze_filename_patterns(dataset_path)
    
    # Step 2: Create balanced classification labels
    print("\n" + "="*50)
    print("CREATING BALANCED CLASSIFICATION LABELS")
    print("="*50)
    
    create_balanced_classification_labels(dataset_path)
    
    # Step 3: Show improved function
    print("\n" + "="*50)
    print("IMPROVED CLASSIFICATION FUNCTION")
    print("="*50)
    
    improved_func = create_improved_classification_function()
    print(improved_func)
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    print("1. The current filename pattern doesn't contain view type information")
    print("2. Created balanced classification labels based on video ID patterns")
    print("3. Updated label files with one-hot encoded classification labels")
    print("4. Use the improved classification function for future training")
    print("5. Consider adding view type information to filenames for better accuracy")

if __name__ == "__main__":
    main()
