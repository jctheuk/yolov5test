#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unify Label Format to V1 Style
Remove blank lines from V2-V5 labels to match V1 format (2 lines only)
"""

import os
from pathlib import Path

def convert_label_format(file_path):
    """
    Convert label file from V2-V5 format (3 lines with blank) to V1 format (2 lines)
    
    V2-V5 format:
    1 0.562297 0.657529 0.160834 0.120714
    
    1 0 0
    
    V1 format:
    1 0.562297 0.657529 0.160834 0.120714
    1 0 0
    """
    
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines and strip whitespace
        non_empty_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only keep non-empty lines
                non_empty_lines.append(stripped_line)
        
        # Write back with V1 format (no blank lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in non_empty_lines:
                f.write(line + '\n')
        
        return len(lines), len(non_empty_lines)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def process_dataset(dataset_path, dataset_name):
    """Process all label files in a dataset"""
    
    print(f"\n--- Processing {dataset_name} ---")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return 0, 0
    
    total_files = 0
    total_converted = 0
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"  {split}: Directory not found")
            continue
        
        # Get all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        split_converted = 0
        
        print(f"  {split}: Processing {len(label_files)} files...")
        
        for label_file in label_files:
            file_path = os.path.join(labels_dir, label_file)
            original_lines, new_lines = convert_label_format(file_path)
            
            if original_lines is not None and new_lines is not None:
                total_files += 1
                if original_lines != new_lines:
                    split_converted += 1
                    total_converted += 1
        
        print(f"    Converted: {split_converted}/{len(label_files)} files")
    
    return total_files, total_converted

def verify_format_consistency():
    """Verify that all datasets now have consistent format"""
    
    print(f"\n{'='*80}")
    print("FORMAT VERIFICATION")
    print("="*80)
    
    datasets = {
        'V1': './regurgitationV1',
        'V2': './regurgitationV2', 
        'V3': './regurgitationV3',
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    # Sample a few files to check format
    sample_files = []
    
    # Get sample files from V1
    v1_train_labels = './regurgitationV1/train/labels'
    if os.path.exists(v1_train_labels):
        files = [f for f in os.listdir(v1_train_labels) if f.endswith('.txt')]
        sample_files = files[:3]  # Take first 3 files as samples
    
    print(f"Checking format consistency for {len(sample_files)} sample files:")
    
    for filename in sample_files:
        print(f"\n{filename}:")
        
        for dataset_name, dataset_path in datasets.items():
            # Find file in any split
            file_content = None
            file_split = None
            
            for split in ['train', 'valid', 'test']:
                file_path = os.path.join(dataset_path, split, 'labels', filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        file_content = content
                        file_split = split
                        break
                    except:
                        continue
            
            if file_content:
                lines = file_content.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                print(f"  {dataset_name} ({file_split}): {len(non_empty_lines)} lines")
                
                # Check for blank lines
                has_blank = len(lines) != len(non_empty_lines)
                if has_blank:
                    print(f"    [WARNING] Contains blank lines!")
                else:
                    print(f"    [OK] No blank lines")
            else:
                print(f"  {dataset_name}: File not found")

def show_format_examples():
    """Show before/after format examples"""
    
    print(f"\n{'='*80}")
    print("FORMAT EXAMPLES")
    print("="*80)
    
    print("BEFORE (V2-V5 format with blank line):")
    print("```")
    print("1 0.562297 0.657529 0.160834 0.120714")
    print("")
    print("1 0 0")
    print("```")
    
    print("\nAFTER (V1 format without blank line):")
    print("```")
    print("1 0.562297 0.657529 0.160834 0.120714")
    print("1 0 0")
    print("```")
    
    print("\nThis change:")
    print("- Removes empty/blank lines")
    print("- Keeps only non-empty content lines")
    print("- Makes all datasets consistent with V1 format")
    print("- Does NOT change detection coordinates or classification labels")

def main():
    """Main function to unify label formats"""
    
    print("LABEL FORMAT UNIFICATION TOOL")
    print("Converting V2-V5 labels to match V1 format (remove blank lines)")
    print("="*80)
    
    # Show format examples
    show_format_examples()
    
    datasets_to_process = {
        'V2': './regurgitationV2',
        'V3': './regurgitationV3', 
        'V4': './regurgitationV4',
        'V5': './regurgitationV5'
    }
    
    total_files_processed = 0
    total_files_converted = 0
    
    print(f"\n{'='*80}")
    print("PROCESSING DATASETS")
    print("="*80)
    
    for dataset_name, dataset_path in datasets_to_process.items():
        files_processed, files_converted = process_dataset(dataset_path, dataset_name)
        total_files_processed += files_processed
        total_files_converted += files_converted
    
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print("="*80)
    
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files converted: {total_files_converted}")
    print(f"Files already in correct format: {total_files_processed - total_files_converted}")
    
    if total_files_converted > 0:
        print(f"\n[SUCCESS] Converted {total_files_converted} files to V1 format!")
    else:
        print(f"\n[INFO] All files were already in correct format")
    
    # Verify format consistency
    verify_format_consistency()

if __name__ == "__main__":
    main()
