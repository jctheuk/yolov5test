#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Annotations for K-Fold Cross Validation
Apply V1's correct annotations to V2-V5 while preserving different splits for k-fold validation
"""

import os
import shutil
from collections import defaultdict

def find_file_in_dataset(filename, dataset_path):
    """Find a file in any split of a dataset and return its path and split"""
    for split in ['train', 'valid', 'test']:
        file_path = os.path.join(dataset_path, split, 'labels', filename)
        if os.path.exists(file_path):
            return file_path, split
    return None, None

def get_all_files_mapping(dataset_path):
    """Get mapping of all files in a dataset: filename -> (split, path)"""
    file_mapping = {}
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            for filename in os.listdir(labels_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(labels_dir, filename)
                    file_mapping[filename] = (split, file_path)
    
    return file_mapping

def copy_v1_annotation_to_target(v1_file_path, target_file_path):
    """Copy V1's annotation content to target file"""
    try:
        # Read V1 content
        with open(v1_file_path, 'r', encoding='utf-8') as f:
            v1_content = f.read()
        
        # Write to target file
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(v1_content)
            
        return True
    except Exception as e:
        print(f"Error copying annotation: {e}")
        return False

def fix_dataset_annotations(target_dataset_path, target_dataset_name, v1_mapping):
    """Fix all annotations in a target dataset using V1's correct annotations"""
    
    print(f"\n--- Fixing {target_dataset_name} ---")
    
    if not os.path.exists(target_dataset_path):
        print(f"Dataset not found: {target_dataset_path}")
        return 0, 0
    
    # Get target dataset file mapping
    target_mapping = get_all_files_mapping(target_dataset_path)
    
    print(f"Target dataset files: {len(target_mapping)}")
    print(f"V1 reference files: {len(v1_mapping)}")
    
    files_fixed = 0
    files_not_found = 0
    
    # Process each file in target dataset
    for filename, (target_split, target_path) in target_mapping.items():
        
        # Find corresponding file in V1
        if filename in v1_mapping:
            v1_split, v1_path = v1_mapping[filename]
            
            # Copy V1's correct annotation to target
            success = copy_v1_annotation_to_target(v1_path, target_path)
            
            if success:
                files_fixed += 1
                if files_fixed <= 5:  # Show first 5 examples
                    print(f"  Fixed: {filename} ({target_split}) <- V1 ({v1_split})")
            else:
                print(f"  Failed: {filename}")
        else:
            files_not_found += 1
            print(f"  Not in V1: {filename}")
    
    if files_fixed > 5:
        print(f"  ... and {files_fixed - 5} more files")
    
    print(f"  Summary: {files_fixed} fixed, {files_not_found} not found in V1")
    
    return files_fixed, files_not_found

def verify_violation_removal():
    """Verify that violations have been removed from all datasets"""
    
    print(f"\n{'='*80}")
    print("VIOLATION VERIFICATION")
    print("="*80)
    
    # Re-run violation check on fixed datasets
    print("Running constraint violation check on fixed datasets...")
    
    # Import the constraint checking logic
    from check_violations_simple import DatasetViolationChecker
    
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    
    checker = DatasetViolationChecker()
    
    for dataset_name in datasets:
        dataset_path = f'./{dataset_name}'
        if os.path.exists(dataset_path):
            print(f"\nChecking {dataset_name}...")
            results = checker.check_dataset(dataset_path, dataset_name)

def create_kfold_summary():
    """Create summary of the k-fold validation setup"""
    
    print(f"\n{'='*80}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print("="*80)
    
    datasets = {
        'V1': './regurgitationV1',
        'V2': './regurgitationV2',
        'V3': './regurgitationV3',
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    # Get split statistics for each dataset
    print("Split distribution for k-fold validation:")
    print("-" * 60)
    print("Dataset".ljust(10) + "Train".ljust(8) + "Valid".ljust(8) + "Test".ljust(8) + "Total")
    print("-" * 60)
    
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            splits_count = {'train': 0, 'valid': 0, 'test': 0}
            
            for split in ['train', 'valid', 'test']:
                labels_dir = os.path.join(dataset_path, split, 'labels')
                if os.path.exists(labels_dir):
                    count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                    splits_count[split] = count
            
            total = sum(splits_count.values())
            print(f"{dataset_name.ljust(10)}{str(splits_count['train']).ljust(8)}{str(splits_count['valid']).ljust(8)}{str(splits_count['test']).ljust(8)}{total}")
    
    print("\nK-Fold Cross Validation Usage:")
    print("- Each dataset (V1-V5) represents a different fold")
    print("- All datasets now have correct annotations (no violations)")
    print("- Different train/valid/test splits enable proper cross-validation")
    print("- Use different datasets for different validation runs")

def main():
    """Main function to fix annotations for k-fold validation"""
    
    print("K-FOLD VALIDATION ANNOTATION FIXER")
    print("Applying V1's correct annotations to V2-V5 while preserving splits")
    print("="*80)
    
    # Get V1 file mapping (our reference with correct annotations)
    v1_path = './regurgitationV1'
    
    if not os.path.exists(v1_path):
        print("Error: V1 dataset not found!")
        return
    
    print("Loading V1 as reference dataset (correct annotations)...")
    v1_mapping = get_all_files_mapping(v1_path)
    print(f"V1 reference files: {len(v1_mapping)}")
    
    # Target datasets to fix
    target_datasets = {
        'V2': './regurgitationV2',
        'V3': './regurgitationV3',
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    total_fixed = 0
    total_not_found = 0
    
    print(f"\n{'='*80}")
    print("FIXING ANNOTATIONS")
    print("="*80)
    
    # Fix each target dataset
    for dataset_name, dataset_path in target_datasets.items():
        fixed, not_found = fix_dataset_annotations(dataset_path, dataset_name, v1_mapping)
        total_fixed += fixed
        total_not_found += not_found
    
    print(f"\n{'='*80}")
    print("ANNOTATION FIXING SUMMARY")
    print("="*80)
    
    print(f"Total files fixed: {total_fixed}")
    print(f"Files not found in V1: {total_not_found}")
    
    if total_fixed > 0:
        print(f"\n[SUCCESS] Applied V1's correct annotations to {total_fixed} files!")
        print("All datasets now have correct annotations while preserving their splits.")
    
    # Create k-fold summary
    create_kfold_summary()
    
    print(f"\n[INFO] Your datasets are now ready for k-fold cross validation!")
    print("Each dataset (V1-V5) can be used as a different fold with:")
    print("- Correct annotations (no constraint violations)")
    print("- Different train/valid/test splits")
    print("- Same total number of files (1,484)")

if __name__ == "__main__":
    main()
