#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check V2-V5 Label Consistency
Verify if files with the same name have identical label content across V2-V5 datasets
"""

import os
import json
from collections import defaultdict

def read_label_file_raw(file_path):
    """Read label file content as raw text"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return None

def find_file_in_splits(dataset_path, filename):
    """Find file in any split and return path and split"""
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        file_path = os.path.join(dataset_path, split, 'labels', filename)
        if os.path.exists(file_path):
            return file_path, split
    
    return None, None

def check_v2_v5_consistency():
    """Check if same-named files have identical content across V2-V5"""
    
    print("="*80)
    print("CHECKING V2-V5 LABEL CONSISTENCY")
    print("="*80)
    
    # Load violation files to check
    violation_file = './violation_analysis/regurgitationV2_constraint_violation_filenames.txt'
    
    with open(violation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    violation_files = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            violation_files.append(line)
    
    print(f"Checking {len(violation_files)} violation files for consistency across V2-V5")
    
    datasets = {
        'V2': './regurgitationV2',
        'V3': './regurgitationV3', 
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    identical_count = 0
    different_count = 0
    differences = []
    
    print(f"\nFile-by-file analysis:")
    print("-" * 60)
    
    for i, filename in enumerate(violation_files, 1):
        print(f"\n{i:2d}. {filename}")
        
        # Collect content from all datasets
        file_contents = {}
        file_splits = {}
        
        for dataset_name, dataset_path in datasets.items():
            file_path, split = find_file_in_splits(dataset_path, filename)
            
            if file_path:
                content = read_label_file_raw(file_path)
                file_contents[dataset_name] = content
                file_splits[dataset_name] = split
            else:
                file_contents[dataset_name] = None
                file_splits[dataset_name] = None
        
        # Check if all contents are identical
        available_contents = {k: v for k, v in file_contents.items() if v is not None}
        
        if len(available_contents) < 2:
            print(f"    [SKIP] Not enough datasets have this file")
            continue
        
        # Compare all contents
        reference_content = list(available_contents.values())[0]
        all_identical = all(content == reference_content for content in available_contents.values())
        
        if all_identical:
            print(f"    [IDENTICAL] All datasets have same content")
            identical_count += 1
            
            # Show splits
            splits_info = ', '.join([f"{ds}({split})" for ds, split in file_splits.items() if split])
            print(f"                Splits: {splits_info}")
            
        else:
            print(f"    [DIFFERENT] Content differs between datasets!")
            different_count += 1
            
            # Show differences in detail
            for dataset_name, content in available_contents.items():
                split = file_splits[dataset_name]
                lines = content.split('\n') if content else []
                print(f"      {dataset_name} ({split}): {len(lines)} lines")
                for j, line in enumerate(lines[:2], 1):  # Show first 2 lines
                    print(f"        L{j}: {line}")
                if len(lines) > 2:
                    print(f"        ... ({len(lines)-2} more lines)")
            
            # Record difference for detailed analysis
            differences.append({
                'filename': filename,
                'contents': available_contents,
                'splits': file_splits
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("CONSISTENCY SUMMARY")
    print("="*80)
    
    total_checked = identical_count + different_count
    
    print(f"\nFiles checked: {total_checked}")
    print(f"Identical across V2-V5: {identical_count} ({identical_count/total_checked*100:.1f}%)")
    print(f"Different across V2-V5: {different_count} ({different_count/total_checked*100:.1f}%)")
    
    if different_count == 0:
        print(f"\n[SUCCESS] All violation files have IDENTICAL content across V2-V5!")
        print("This confirms that the conversion process was completely consistent.")
    else:
        print(f"\n[WARNING] Found {different_count} files with different content!")
        print("This suggests inconsistencies in the conversion process.")
        
        # Show detailed differences
        if differences:
            print(f"\nDetailed differences:")
            for diff in differences[:3]:  # Show first 3 differences
                filename = diff['filename']
                print(f"\n  {filename}:")
                for dataset, content in diff['contents'].items():
                    lines = content.split('\n')
                    print(f"    {dataset}: {' | '.join(lines)}")
    
    return identical_count, different_count, differences

def check_all_files_consistency():
    """Check consistency for ALL files, not just violations"""
    
    print(f"\n{'='*80}")
    print("CHECKING ALL FILES CONSISTENCY (SAMPLE)")
    print("="*80)
    
    # Sample some files from V2 to check consistency
    v2_train_labels = './regurgitationV2/train/labels'
    
    if not os.path.exists(v2_train_labels):
        print("V2 train labels directory not found")
        return
    
    # Get sample of files from V2
    all_files = [f for f in os.listdir(v2_train_labels) if f.endswith('.txt')]
    sample_files = all_files[::50]  # Every 50th file as sample
    
    print(f"Checking {len(sample_files)} sample files from all datasets")
    
    datasets = {
        'V2': './regurgitationV2',
        'V3': './regurgitationV3', 
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    identical_count = 0
    different_count = 0
    
    for i, filename in enumerate(sample_files, 1):
        print(f"\n{i:2d}. {filename}")
        
        # Collect content from all datasets
        file_contents = {}
        
        for dataset_name, dataset_path in datasets.items():
            file_path, split = find_file_in_splits(dataset_path, filename)
            
            if file_path:
                content = read_label_file_raw(file_path)
                file_contents[dataset_name] = content
        
        # Check consistency
        available_contents = {k: v for k, v in file_contents.items() if v is not None}
        
        if len(available_contents) >= 2:
            reference_content = list(available_contents.values())[0]
            all_identical = all(content == reference_content for content in available_contents.values())
            
            if all_identical:
                print(f"    [IDENTICAL]")
                identical_count += 1
            else:
                print(f"    [DIFFERENT]")
                different_count += 1
                
                # Show first difference found
                contents_list = list(available_contents.items())
                if len(contents_list) >= 2:
                    ds1, content1 = contents_list[0]
                    ds2, content2 = contents_list[1]
                    print(f"      {ds1}: {content1[:50]}...")
                    print(f"      {ds2}: {content2[:50]}...")
    
    total = identical_count + different_count
    if total > 0:
        print(f"\nSample consistency check:")
        print(f"  Identical: {identical_count}/{total} ({identical_count/total*100:.1f}%)")
        print(f"  Different: {different_count}/{total} ({different_count/total*100:.1f}%)")

if __name__ == "__main__":
    # Check violation files consistency
    identical, different, differences = check_v2_v5_consistency()
    
    # Check sample of all files
    check_all_files_consistency()
