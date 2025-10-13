#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Comparison Data
Get violation filenames, corresponding images, and V1 reference data for manual comparison
"""

import os
import json
from pathlib import Path

def find_file_in_dataset(filename, dataset_path, file_type='labels'):
    """Find file in dataset splits and return path and split name"""
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        if file_type == 'labels':
            file_path = os.path.join(dataset_path, split, 'labels', filename)
        else:  # images
            # Convert .txt to image extension
            image_filename = filename.replace('.txt', '.png')  # Assuming .png, might be .jpg
            file_path = os.path.join(dataset_path, split, 'images', image_filename)
        
        if os.path.exists(file_path):
            return file_path, split
    
    return None, None

def read_label_content(file_path):
    """Read and parse label file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        lines = content.split('\n')
        detections = []
        classification = None
        
        # Detection and view class names
        detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
        view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # Classification line (3 elements, all 0 or 1)
            if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                for i, val in enumerate(parts):
                    if val == '1':
                        classification = i
                        break
            
            # Detection line (5+ elements)
            elif len(parts) >= 5:
                try:
                    detection_class = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]  # x, y, w, h
                    detections.append({
                        'class': detection_class,
                        'name': detection_names.get(detection_class, f'Unknown({detection_class})'),
                        'bbox': bbox
                    })
                except:
                    continue
        
        return {
            'raw_content': content,
            'detections': detections,
            'classification': classification,
            'classification_name': view_names.get(classification, f'Unknown({classification})')
        }
    except:
        return None

def prepare_comparison_data():
    """Prepare all comparison data for manual review"""
    
    print("="*80)
    print("PREPARING COMPARISON DATA FOR MANUAL REVIEW")
    print("="*80)
    
    # Load violation files from V2
    violation_file = './violation_analysis/regurgitationV2_constraint_violation_filenames.txt'
    
    if not os.path.exists(violation_file):
        print(f"Error: {violation_file} not found")
        print("Please run the violation analysis first")
        return
    
    # Read violation files
    with open(violation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    violation_files = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            violation_files.append(line)
    
    print(f"Found {len(violation_files)} violation files to compare")
    
    # Prepare comparison data
    comparison_data = []
    
    datasets = {
        'V1': './regurgitationV1',
        'V2': './regurgitationV2', 
        'V3': './regurgitationV3',
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    print(f"\nProcessing files...")
    print("-" * 50)
    
    for i, filename in enumerate(violation_files, 1):
        print(f"\n{i:2d}. {filename}")
        
        file_info = {
            'filename': filename,
            'image_filename': filename.replace('.txt', '.png'),
            'datasets': {}
        }
        
        # Check each dataset
        for dataset_name, dataset_path in datasets.items():
            
            # Find label file
            label_path, label_split = find_file_in_dataset(filename, dataset_path, 'labels')
            
            # Find image file  
            image_path, image_split = find_file_in_dataset(filename, dataset_path, 'images')
            
            if label_path:
                label_data = read_label_content(label_path)
                
                file_info['datasets'][dataset_name] = {
                    'label_path': label_path,
                    'label_split': label_split,
                    'label_data': label_data,
                    'image_path': image_path,
                    'image_split': image_split,
                    'found': True
                }
                
                if label_data:
                    detection_summary = ', '.join([d['name'] for d in label_data['detections']])
                    print(f"   {dataset_name} ({label_split}): {detection_summary} + {label_data['classification_name']}")
                else:
                    print(f"   {dataset_name} ({label_split}): [Parse Error]")
            else:
                file_info['datasets'][dataset_name] = {
                    'found': False
                }
                print(f"   {dataset_name}: [NOT FOUND]")
        
        comparison_data.append(file_info)
    
    # Save detailed comparison data
    output_file = './violation_analysis/comparison_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'-'*80}")
    print("COMPARISON SUMMARY")
    print("-"*80)
    
    # Generate human-readable comparison report
    create_human_readable_report(comparison_data)
    
    print(f"\nDetailed data saved to: {output_file}")
    print(f"Human-readable report saved to: ./violation_analysis/comparison_report.md")

def create_human_readable_report(comparison_data):
    """Create a human-readable comparison report"""
    
    report_file = './violation_analysis/comparison_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Manual Comparison Report\n\n")
        f.write("This report shows violation files and their corresponding data across datasets.\n")
        f.write("Use this to manually compare images and annotations.\n\n")
        f.write("---\n\n")
        
        for i, file_info in enumerate(comparison_data, 1):
            filename = file_info['filename']
            image_filename = file_info['image_filename']
            
            f.write(f"## {i}. {filename}\n\n")
            f.write(f"**Image file**: `{image_filename}`\n\n")
            
            # Create comparison table
            f.write("| Dataset | Split | Detections | View | Image Path | Label Path |\n")
            f.write("|---------|-------|------------|------|------------|------------|\n")
            
            for dataset_name in ['V1', 'V2', 'V3', 'V4', 'V5']:
                dataset_info = file_info['datasets'].get(dataset_name, {})
                
                if dataset_info.get('found', False):
                    label_data = dataset_info.get('label_data', {})
                    detections = ', '.join([d['name'] for d in label_data.get('detections', [])])
                    view = label_data.get('classification_name', 'Unknown')
                    split = dataset_info.get('label_split', 'Unknown')
                    image_path = dataset_info.get('image_path', 'Not found')
                    label_path = dataset_info.get('label_path', 'Not found')
                    
                    f.write(f"| **{dataset_name}** | {split} | {detections} | {view} | `{image_path}` | `{label_path}` |\n")
                else:
                    f.write(f"| **{dataset_name}** | - | NOT FOUND | - | - | - |\n")
            
            f.write("\n### Raw Label Contents\n\n")
            
            # Show raw content for comparison
            for dataset_name in ['V1', 'V2']:  # Focus on V1 vs V2 for main comparison
                dataset_info = file_info['datasets'].get(dataset_name, {})
                
                if dataset_info.get('found', False):
                    label_data = dataset_info.get('label_data', {})
                    raw_content = label_data.get('raw_content', '')
                    
                    f.write(f"**{dataset_name}**:\n```\n{raw_content}\n```\n\n")
                else:
                    f.write(f"**{dataset_name}**: NOT FOUND\n\n")
            
            f.write("### Instructions for Manual Review\n\n")
            f.write("1. Open the image file in an image viewer\n")
            f.write("2. Compare the annotations from different datasets\n") 
            f.write("3. Determine which annotation is medically correct\n")
            f.write("4. Note your decision for batch correction\n\n")
            f.write("---\n\n")
    
    print(f"Human-readable report created with {len(comparison_data)} files")

def create_image_copy_script(comparison_data):
    """Create script to copy images for easier review"""
    
    script_file = './violation_analysis/copy_images_for_review.py'
    
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
# Auto-generated script to copy violation images for review

import os
import shutil

def copy_images_for_review():
    """Copy all violation images to a review folder"""
    
    review_dir = './violation_analysis/images_for_review'
    os.makedirs(review_dir, exist_ok=True)
    
    image_paths = [
''')
        
        # Add image paths
        for file_info in comparison_data:
            for dataset_name, dataset_info in file_info['datasets'].items():
                if dataset_info.get('found') and dataset_info.get('image_path'):
                    image_path = dataset_info['image_path']
                    filename = file_info['filename']
                    f.write(f"        ('{image_path}', '{dataset_name}_{filename.replace('.txt', '.png')}'),\n")
        
        f.write('''    ]
    
    for src_path, dst_name in image_paths:
        if os.path.exists(src_path):
            dst_path = os.path.join(review_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {dst_name}")
        else:
            print(f"Not found: {src_path}")
    
    print(f"\\nImages copied to: {review_dir}")

if __name__ == "__main__":
    copy_images_for_review()
''')
    
    print(f"Image copy script created: {script_file}")
    print("Run 'python violation_analysis/copy_images_for_review.py' to copy images for review")

if __name__ == "__main__":
    prepare_comparison_data()
