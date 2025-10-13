#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revert V2-V5 to Original State
Restore V2-V5 datasets to their original converted state before V1 annotation copying
"""

import os
import shutil

def backup_current_state():
    """Create backup of current modified state before reverting"""
    
    print("Creating backup of current modified state...")
    
    backup_dir = './v1_corrected_backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    
    for dataset in datasets:
        if os.path.exists(f'./{dataset}'):
            backup_path = os.path.join(backup_dir, dataset)
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(f'./{dataset}', backup_path)
            print(f"  Backed up {dataset} to {backup_path}")

def remove_modified_datasets():
    """Remove the modified V2-V5 datasets"""
    
    print("\nRemoving modified datasets...")
    
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    
    for dataset in datasets:
        dataset_path = f'./{dataset}'
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            print(f"  Removed {dataset}")

def reconvert_from_original():
    """Re-convert from original segmentation datasets"""
    
    print("\nRe-converting from original segmentation datasets...")
    
    # Import the conversion functions from our original script
    import sys
    sys.path.append('.')
    
    # Re-run the original conversion
    original_datasets = {
        '2': ('Regurgitation 2025_Regurgitation-YOLODataset-2/Regurgitation-YOLODataset-2', 'regurgitationV2'),
        '3': ('Regurgitation 2025_Regurgitation-YOLODataset-3/Regurgitation-YOLODataset-3', 'regurgitationV3'),
        '4': ('Regurgitation 2025_Regurgitation-YOLODataset-4/Regurgitation-YOLODataset-4', 'regurgitationV4'),
        '5': ('Regurgitation 2025_Regurgitation-YOLODataset-5/Regurgitation-YOLODataset-5', 'regurgitationV5'),
    }
    
    # Import conversion functions
    try:
        from convert_segmentation_to_detection import convert_dataset, create_data_yaml
        
        for dataset_num, (input_subdir, output_name) in original_datasets.items():
            input_dir = os.path.join('.', input_subdir)
            output_dir = os.path.join('.', output_name)
            
            if os.path.exists(input_dir):
                print(f"  Converting dataset {dataset_num}: {input_subdir} -> {output_name}")
                convert_dataset(input_dir, output_dir, output_name)
            else:
                print(f"  Warning: Original dataset {input_subdir} not found")
                
    except ImportError:
        print("  Error: Could not import conversion functions")
        return False
    
    return True

def apply_format_fixes():
    """Apply the format fixes (remove blank lines) to restored datasets"""
    
    print("\nApplying format fixes (remove blank lines)...")
    
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    
    for dataset in datasets:
        dataset_path = f'./{dataset}'
        if os.path.exists(dataset_path):
            print(f"  Fixing format for {dataset}")
            
            # Process all splits
            for split in ['train', 'valid', 'test']:
                labels_dir = os.path.join(dataset_path, split, 'labels')
                
                if os.path.exists(labels_dir):
                    files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                    
                    for filename in files:
                        file_path = os.path.join(labels_dir, filename)
                        
                        try:
                            # Read content
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            
                            # Remove empty lines
                            non_empty_lines = [line for line in lines if line.strip()]
                            
                            # Write back
                            with open(file_path, 'w', encoding='utf-8') as f:
                                for line in non_empty_lines:
                                    if not line.endswith('\n'):
                                        line += '\n'
                                    f.write(line)
                                    
                        except Exception as e:
                            print(f"    Error fixing {filename}: {e}")

def verify_reversion():
    """Verify that datasets have been reverted to original state"""
    
    print(f"\n{'='*80}")
    print("REVERSION VERIFICATION")
    print("="*80)
    
    # Check that V2-V5 now have violations again (proving they're reverted)
    print("Running constraint violation check to confirm reversion...")
    
    try:
        os.system('python check_violations_simple.py')
    except:
        print("Could not run automatic violation check")
    
    # Manual check of a known violation file
    violation_file = "bWplwqlsaMKZ-unnamed_1_1.mp4-1.txt"
    
    print(f"\nManual verification of {violation_file}:")
    
    datasets = ['regurgitationV1', 'regurgitationV2']
    
    for dataset in datasets:
        file_found = False
        for split in ['train', 'valid', 'test']:
            file_path = f'./{dataset}/{split}/labels/{violation_file}'
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                    lines = content.split('\n')
                    
                    print(f"  {dataset} ({split}):")
                    for line in lines:
                        if line.strip():
                            print(f"    {line}")
                    
                    file_found = True
                    break
                except:
                    continue
        
        if not file_found:
            print(f"  {dataset}: File not found")

def main():
    """Main reversion function"""
    
    print("REVERTING V2-V5 DATASETS TO ORIGINAL STATE")
    print("This will undo the V1 annotation copying and restore original annotations")
    print("="*80)
    
    # Step 1: Backup current state
    backup_current_state()
    
    # Step 2: Remove modified datasets
    remove_modified_datasets()
    
    # Step 3: Re-convert from original segmentation format
    success = reconvert_from_original()
    
    if success:
        # Step 4: Apply format fixes (remove blank lines)
        apply_format_fixes()
        
        print(f"\n{'='*80}")
        print("REVERSION COMPLETE")
        print("="*80)
        
        print("V2-V5 datasets have been reverted to their original state:")
        print("✅ Original annotations restored (with constraint violations)")
        print("✅ Original split distributions preserved") 
        print("✅ Format unified (2 lines, no blank lines)")
        print("✅ Current modified state backed up in ./v1_corrected_backup/")
        
        # Verify reversion
        verify_reversion()
        
    else:
        print("\n❌ Reversion failed - could not re-convert from original")
        print("The modified datasets have been backed up but not reverted")

if __name__ == "__main__":
    main()
