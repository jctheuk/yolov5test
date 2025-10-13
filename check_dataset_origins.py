#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Dataset Origins
Verify if V1 and V2-V5 come from the same original dataset with different splits
"""

import os
from collections import defaultdict, Counter

def get_all_files_from_dataset(dataset_path):
    """Get all label files from a dataset across all splits"""
    all_files = set()
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            all_files.update(files)
    
    return all_files

def get_file_split_mapping(dataset_path):
    """Get mapping of filename -> split for a dataset"""
    file_splits = {}
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            for filename in files:
                file_splits[filename] = split
    
    return file_splits

def analyze_dataset_origins():
    """Analyze if all datasets come from the same original source"""
    
    print("="*80)
    print("DATASET ORIGINS ANALYSIS")
    print("="*80)
    
    datasets = {
        'V1': './regurgitationV1',
        'V2': './regurgitationV2',
        'V3': './regurgitationV3',
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    # Get all files from each dataset
    dataset_files = {}
    dataset_splits = {}
    
    for name, path in datasets.items():
        if os.path.exists(path):
            dataset_files[name] = get_all_files_from_dataset(path)
            dataset_splits[name] = get_file_split_mapping(path)
            print(f"{name}: {len(dataset_files[name])} files")
        else:
            print(f"{name}: NOT FOUND")
            dataset_files[name] = set()
    
    # Analyze file overlap
    print(f"\n" + "-"*60)
    print("FILE OVERLAP ANALYSIS")
    print("-"*60)
    
    # Find common files across all datasets
    all_datasets = [name for name in datasets.keys() if dataset_files[name]]
    
    if len(all_datasets) >= 2:
        # Start with first dataset, find intersection with others
        common_files = dataset_files[all_datasets[0]].copy()
        
        for dataset_name in all_datasets[1:]:
            common_files &= dataset_files[dataset_name]
            print(f"{all_datasets[0]} & {dataset_name}: {len(common_files)} common files")
    
    print(f"\nFiles common to ALL datasets: {len(common_files)}")
    
    # Analyze unique files in each dataset
    print(f"\nUNIQUE FILES ANALYSIS:")
    
    for dataset_name in all_datasets:
        unique_to_this = dataset_files[dataset_name].copy()
        for other_name in all_datasets:
            if other_name != dataset_name:
                unique_to_this -= dataset_files[other_name]
        
        print(f"{dataset_name} unique files: {len(unique_to_this)}")
        if len(unique_to_this) > 0 and len(unique_to_this) <= 5:
            print(f"  Examples: {list(unique_to_this)[:3]}")
    
    # Check if V1 is a subset of others (due to cleaning)
    print(f"\nSUBSET ANALYSIS:")
    v1_files = dataset_files.get('V1', set())
    v2_files = dataset_files.get('V2', set())
    
    if v1_files and v2_files:
        v1_subset_of_v2 = v1_files.issubset(v2_files)
        v2_subset_of_v1 = v2_files.issubset(v1_files)
        
        print(f"V1 is subset of V2: {v1_subset_of_v2}")
        print(f"V2 is subset of V1: {v2_subset_of_v1}")
        
        if v1_subset_of_v2:
            missing_in_v1 = v2_files - v1_files
            print(f"Files in V2 but not V1: {len(missing_in_v1)}")
        
        if v2_subset_of_v1:
            missing_in_v2 = v1_files - v2_files
            print(f"Files in V1 but not V2: {len(missing_in_v2)}")
    
    return common_files, dataset_files, dataset_splits

def analyze_split_distributions(common_files, dataset_splits):
    """Analyze how common files are distributed across splits in different datasets"""
    
    print(f"\n" + "="*80)
    print("SPLIT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    if not common_files:
        print("No common files to analyze")
        return
    
    print(f"Analyzing split distributions for {len(common_files)} common files")
    
    # Sample some files for detailed analysis
    sample_files = list(common_files)[:10]
    
    print(f"\nSample file split distributions:")
    print("-"*60)
    
    for i, filename in enumerate(sample_files, 1):
        print(f"\n{i}. {filename}")
        
        split_info = {}
        for dataset_name, file_splits in dataset_splits.items():
            split_info[dataset_name] = file_splits.get(filename, 'NOT_FOUND')
        
        # Show split distribution
        splits_str = ' | '.join([f"{ds}: {split}" for ds, split in split_info.items()])
        print(f"   {splits_str}")
        
        # Check if splits are different
        unique_splits = set(split_info.values())
        unique_splits.discard('NOT_FOUND')
        
        if len(unique_splits) > 1:
            print(f"   [DIFFERENT] File in different splits across datasets")
        elif len(unique_splits) == 1:
            print(f"   [SAME] File in same split ({list(unique_splits)[0]}) across all datasets")
    
    # Overall split distribution statistics
    print(f"\n" + "-"*60)
    print("OVERALL SPLIT DISTRIBUTION STATISTICS")
    print("-"*60)
    
    for dataset_name, file_splits in dataset_splits.items():
        if not file_splits:
            continue
            
        split_counts = Counter(file_splits.values())
        total = sum(split_counts.values())
        
        print(f"\n{dataset_name} ({total} files):")
        for split, count in sorted(split_counts.items()):
            percentage = count/total*100
            print(f"  {split}: {count} ({percentage:.1f}%)")

def check_label_content_differences(common_files, dataset_splits):
    """Check if same files have different label content across datasets"""
    
    print(f"\n" + "="*80)
    print("LABEL CONTENT COMPARISON")
    print("="*80)
    
    # Sample files for content comparison
    sample_files = list(common_files)[:5]
    
    datasets = {
        'V1': './regurgitationV1',
        'V2': './regurgitationV2'
    }
    
    different_content_count = 0
    same_content_count = 0
    
    for filename in sample_files:
        print(f"\n{filename}:")
        
        file_contents = {}
        
        for dataset_name, dataset_path in datasets.items():
            splits = dataset_splits.get(dataset_name, {})
            split = splits.get(filename)
            
            if split:
                file_path = os.path.join(dataset_path, split, 'labels', filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    file_contents[dataset_name] = content
                    
                    lines = content.split('\n')
                    print(f"  {dataset_name} ({split}): {len(lines)} lines")
                    for line in lines[:2]:  # Show first 2 lines
                        print(f"    {line}")
                        
                except Exception as e:
                    print(f"  {dataset_name}: Error reading file")
        
        # Compare content
        if len(file_contents) >= 2:
            contents_list = list(file_contents.values())
            if len(set(contents_list)) == 1:
                print(f"  [SAME CONTENT]")
                same_content_count += 1
            else:
                print(f"  [DIFFERENT CONTENT]")
                different_content_count += 1
    
    print(f"\nContent comparison summary:")
    print(f"  Same content: {same_content_count}")
    print(f"  Different content: {different_content_count}")

if __name__ == "__main__":
    # Analyze dataset origins
    common_files, dataset_files, dataset_splits = analyze_dataset_origins()
    
    # Analyze split distributions
    analyze_split_distributions(common_files, dataset_splits)
    
    # Check label content differences
    check_label_content_differences(common_files, dataset_splits)
