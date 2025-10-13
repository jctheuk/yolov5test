#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check V1 Dataset Constraint Violations
Verify if regurgitationV1 has been properly cleaned
"""

import os
from collections import defaultdict

class AnatomicalConstraints:
    """Anatomical constraint definitions"""
    
    def __init__(self):
        # View class definitions
        self.view_names = {
            0: 'A4C',   # Apical 4-Chamber
            1: 'PSAX',  # Parasternal Short Axis  
            2: 'PLAX'   # Parasternal Long Axis
        }
        
        # Regurgitation class definitions
        self.regurg_names = {
            0: 'AR',    # Aortic Regurgitation
            1: 'MR',    # Mitral Regurgitation
            2: 'PR',    # Pulmonary Regurgitation
            3: 'TR'     # Tricuspid Regurgitation
        }
        
        # Anatomical constraint rules: allowed regurgitation types per view
        self.constraints = {
            0: [1, 3],  # A4C: only allows MR (1), TR (3)
            1: [2, 3],  # PSAX: only allows PR (2), TR (3)
            2: [0, 1],  # PLAX: only allows AR (0), MR (1)
        }
    
    def is_violation(self, view_class, detection_class):
        """Check if detection violates anatomical constraints"""
        if view_class not in self.constraints:
            return False
        allowed_classes = self.constraints[view_class]
        return detection_class not in allowed_classes
    
    def get_violation_type(self, view_class, detection_class):
        """Get violation type description"""
        if not self.is_violation(view_class, detection_class):
            return None
        view_name = self.view_names.get(view_class, f'VIEW_{view_class}')
        regurg_name = self.regurg_names.get(detection_class, f'REGURG_{detection_class}')
        return f"{view_name}_{regurg_name}"


def parse_label_file(label_path):
    """Parse label file and extract detections and view class"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) < 2:
            return None, None
        
        detections = []
        view_class = None
        
        for line in lines:
            parts = line.split()
            
            # Check if this is classification line (3 elements, all 0 or 1)
            if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                # Find which position has '1' to get view class
                for i, val in enumerate(parts):
                    if val == '1':
                        view_class = i
                        break
                    
            # Check if this is detection line (5+ elements)
            elif len(parts) >= 5:
                try:
                    detection_class = int(parts[0])
                    detections.append(detection_class)
                except ValueError:
                    continue
        
        return detections, view_class
        
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
        return None, None


def check_v1_dataset():
    """Check regurgitationV1 dataset for constraint violations"""
    
    print("="*60)
    print("Checking regurgitationV1 Dataset for Constraint Violations")
    print("="*60)
    
    constraints = AnatomicalConstraints()
    dataset_path = './regurgitationV1'
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return
    
    results = {
        'total_files': 0,
        'parsed_files': 0,
        'violation_files': 0,
        'violations_by_type': defaultdict(int),
        'violations_by_split': defaultdict(int),
        'violation_details': []
    }
    
    # Check each split (train, valid, test)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"  Warning: {labels_dir} not found, skipping...")
            continue
        
        print(f"\n--- Checking {split} split ---")
        
        split_files = 0
        split_violations = 0
        split_violation_details = []
        
        # Process all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            results['total_files'] += 1
            split_files += 1
            
            # Parse label file
            detections, view_class = parse_label_file(label_path)
            
            if detections is None or view_class is None:
                continue
            
            results['parsed_files'] += 1
            
            # Check each detection for violations
            file_violations = []
            
            for detection_class in detections:
                if constraints.is_violation(view_class, detection_class):
                    violation_type = constraints.get_violation_type(view_class, detection_class)
                    
                    violation_info = {
                        'file': label_file,
                        'split': split,
                        'view_class': view_class,
                        'view_name': constraints.view_names[view_class],
                        'detection_class': detection_class,
                        'detection_name': constraints.regurg_names[detection_class],
                        'violation_type': violation_type
                    }
                    
                    file_violations.append(violation_info)
                    results['violations_by_type'][violation_type] += 1
                    split_violation_details.append(violation_info)
            
            # Record file if it has violations
            if file_violations:
                results['violation_files'] += 1
                results['violations_by_split'][split] += 1
                split_violations += 1
                results['violation_details'].extend(file_violations)
        
        print(f"  Files processed: {split_files}")
        print(f"  Violations found: {split_violations}")
        
        # Show violation details for this split
        if split_violation_details:
            print(f"  Violation details:")
            for violation in split_violation_details[:5]:  # Show first 5
                print(f"    {violation['file']}: {violation['violation_type']}")
            if len(split_violation_details) > 5:
                print(f"    ... and {len(split_violation_details) - 5} more")
    
    # Final summary
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if results['parsed_files'] > 0:
        violation_rate = (results['violation_files'] / results['parsed_files']) * 100
        print(f"Total files: {results['total_files']}")
        print(f"Successfully parsed: {results['parsed_files']}")
        print(f"Violation files: {results['violation_files']}")
        print(f"Violation rate: {violation_rate:.2f}%")
        
        if results['violation_files'] == 0:
            print("\nSUCCESS: No constraint violations found!")
            print("The regurgitationV1 dataset is clean and ready for training.")
        else:
            print(f"\nWARNING: Found {results['violation_files']} violation files")
            print("Violation type breakdown:")
            for vtype, count in sorted(results['violations_by_type'].items()):
                percentage = (count / len(results['violation_details'])) * 100
                print(f"  {vtype}: {count} ({percentage:.1f}%)")
    else:
        print("ERROR: No files could be parsed")
    
    return results


if __name__ == "__main__":
    results = check_v1_dataset()
