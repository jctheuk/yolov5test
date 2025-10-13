#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove Corrupted File from V2-V5
Find and remove the file that exists in V2-V5 but not in V1 (corrupted file)
"""

import os
import shutil

def get_all_files_from_dataset(dataset_path):
    """Get all label files from a dataset across all splits"""
    all_files = set()
    file_paths = {}  # filename -> (split, full_path)
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            for filename in files:
                all_files.add(filename)
                file_paths[filename] = (split, os.path.join(labels_dir, filename))
    
    return all_files, file_paths

def find_corrupted_file():
    """Find the file that exists in V2-V5 but not in V1"""
    
    print("="*80)
    print("FINDING CORRUPTED FILE")
    print("="*80)
    
    # Get files from V1 and V2
    v1_files, v1_paths = get_all_files_from_dataset('./regurgitationV1')
    v2_files, v2_paths = get_all_files_from_dataset('./regurgitationV2')
    
    print(f"V1 files: {len(v1_files)}")
    print(f"V2 files: {len(v2_files)}")
    
    # Find files in V2 but not in V1
    extra_files = v2_files - v1_files
    
    print(f"\nFiles in V2 but not V1: {len(extra_files)}")
    
    if extra_files:
        corrupted_file = list(extra_files)[0]
        print(f"Corrupted file found: {corrupted_file}")
        
        # Find where this file exists in V2
        if corrupted_file in v2_paths:
            split, file_path = v2_paths[corrupted_file]
            print(f"Location in V2: {split} - {file_path}")
        
        return corrupted_file
    else:
        print("No extra files found")
        return None

def remove_file_from_dataset(dataset_path, filename):
    """Remove a specific file from all splits in a dataset"""
    
    removed_files = []
    
    for split in ['train', 'valid', 'test']:
        # Remove label file
        label_path = os.path.join(dataset_path, split, 'labels', filename)
        if os.path.exists(label_path):
            os.remove(label_path)
            removed_files.append(f"Label: {label_path}")
        
        # Remove corresponding image file
        image_filename = filename.replace('.txt', '.png')
        image_path = os.path.join(dataset_path, split, 'images', image_filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            removed_files.append(f"Image: {image_path}")
        
        # Try .jpg if .png doesn't exist
        if not os.path.exists(image_path):
            image_filename_jpg = filename.replace('.txt', '.jpg')
            image_path_jpg = os.path.join(dataset_path, split, 'images', image_filename_jpg)
            if os.path.exists(image_path_jpg):
                os.remove(image_path_jpg)
                removed_files.append(f"Image: {image_path_jpg}")
    
    return removed_files

def remove_corrupted_file_from_all():
    """Remove the corrupted file from all V2-V5 datasets"""
    
    # Find the corrupted file
    corrupted_file = find_corrupted_file()
    
    if not corrupted_file:
        print("No corrupted file found to remove")
        return
    
    print(f"\n{'='*80}")
    print("REMOVING CORRUPTED FILE FROM ALL DATASETS")
    print("="*80)
    
    datasets = {
        'V2': './regurgitationV2',
        'V3': './regurgitationV3', 
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    total_removed = 0
    
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            print(f"\n--- Removing from {dataset_name} ---")
            
            removed_files = remove_file_from_dataset(dataset_path, corrupted_file)
            
            if removed_files:
                print(f"Removed {len(removed_files)} files:")
                for removed_file in removed_files:
                    print(f"  - {removed_file}")
                total_removed += len(removed_files)
            else:
                print("  No files found to remove")
        else:
            print(f"{dataset_name}: Dataset not found")
    
    print(f"\n{'='*80}")
    print("REMOVAL SUMMARY")
    print("="*80)
    
    print(f"Corrupted file: {corrupted_file}")
    print(f"Total files removed: {total_removed}")
    
    # Verify removal by checking file counts
    print(f"\nVerification - File counts after removal:")
    
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            files, _ = get_all_files_from_dataset(dataset_path)
            print(f"{dataset_name}: {len(files)} files")
    
    # Check if all datasets now have same count as V1
    v1_files, _ = get_all_files_from_dataset('./regurgitationV1')
    v1_count = len(v1_files)
    
    print(f"V1 (reference): {v1_count} files")
    
    all_match = True
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            files, _ = get_all_files_from_dataset(dataset_path)
            if len(files) != v1_count:
                all_match = False
                print(f"[WARNING] {dataset_name} still has {len(files)} files, expected {v1_count}")
    
    if all_match:
        print(f"\n[SUCCESS] All datasets now have {v1_count} files, matching V1!")
    else:
        print(f"\n[WARNING] Some datasets still don't match V1 count")

def backup_corrupted_file():
    """Create backup of corrupted file before removal"""
    
    corrupted_file = find_corrupted_file()
    
    if not corrupted_file:
        return
    
    print(f"\n--- Creating backup of {corrupted_file} ---")
    
    backup_dir = './corrupted_file_backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup from V2
    v2_files, v2_paths = get_all_files_from_dataset('./regurgitationV2')
    
    if corrupted_file in v2_paths:
        split, source_path = v2_paths[corrupted_file]
        backup_path = os.path.join(backup_dir, f"V2_{split}_{corrupted_file}")
        
        try:
            shutil.copy2(source_path, backup_path)
            print(f"Backed up label: {backup_path}")
            
            # Also backup corresponding image
            image_filename = corrupted_file.replace('.txt', '.png')
            source_image_path = os.path.join('./regurgitationV2', split, 'images', image_filename)
            
            if os.path.exists(source_image_path):
                backup_image_path = os.path.join(backup_dir, f"V2_{split}_{image_filename}")
                shutil.copy2(source_image_path, backup_image_path)
                print(f"Backed up image: {backup_image_path}")
                
        except Exception as e:
            print(f"Error backing up: {e}")

if __name__ == "__main__":
    print("CORRUPTED FILE REMOVAL TOOL")
    print("This will find and remove the file that exists in V2-V5 but not in V1")
    print("="*80)
    
    # Create backup first
    backup_corrupted_file()
    
    # Remove corrupted file
    remove_corrupted_file_from_all()
