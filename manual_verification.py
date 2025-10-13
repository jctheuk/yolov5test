#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Verification of Fixed Violations
Double-check specific files that previously had violations
"""

import os

def check_specific_violations():
    """Check specific files that previously had violations"""
    
    print("="*80)
    print("MANUAL VERIFICATION OF PREVIOUSLY VIOLATED FILES")
    print("="*80)
    
    # These were the main violation files from our earlier analysis
    violation_files = [
        'bWplwqlsaMKZ-unnamed_1_1.mp4-1.txt',     # Was A4C_AR violation
        'ZmZnwqlqbMKawp0=-unnamed_1_1.mp4-15.txt', # Was A4C_PR violation  
        'ZmhmwqduY8KU-Mmode+2D+Doppler_Echo_color_1_2.mp4-1.txt', # Was PLAX_TR violation
        'ZmNlwq5mZcKcwps=-unnamed_1_1.mp4-0.txt', # Was PSAX_AR violation
        'ZmNmwq5saG5m-unnamed_1_3.mp4-8.txt'      # Was PSAX_MR violation
    ]
    
    # Constraint definitions for checking
    constraints = {
        0: [1, 3],  # A4C: allows MR(1), TR(3)
        1: [2, 3],  # PSAX: allows PR(2), TR(3)  
        2: [0, 1],  # PLAX: allows AR(0), MR(1)
    }
    
    detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
    view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
    
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    
    for i, filename in enumerate(violation_files, 1):
        print(f"\n{i}. {filename}")
        print("-" * 60)
        
        for dataset in datasets:
            # Find file in any split
            file_content = None
            file_split = None
            
            for split in ['train', 'valid', 'test']:
                file_path = os.path.join(f'./{dataset}', split, 'labels', filename)
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
                lines = [line.strip() for line in file_content.split('\n') if line.strip()]
                
                # Parse content
                detections = []
                view_class = None
                
                for line in lines:
                    parts = line.split()
                    
                    # Classification line
                    if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                        for j, val in enumerate(parts):
                            if val == '1':
                                view_class = j
                                break
                    
                    # Detection line
                    elif len(parts) >= 5:
                        try:
                            detection_class = int(parts[0])
                            detections.append(detection_class)
                        except:
                            continue
                
                # Check for violations
                violations = []
                if view_class is not None and view_class in constraints:
                    allowed_classes = constraints[view_class]
                    for detection in detections:
                        if detection not in allowed_classes:
                            violations.append(f"{view_names[view_class]}_{detection_names[detection]}")
                
                # Report
                if violations:
                    print(f"  {dataset} ({file_split}): VIOLATION - {', '.join(violations)}")
                else:
                    detection_str = ', '.join([detection_names[d] for d in detections])
                    view_str = view_names.get(view_class, 'Unknown')
                    print(f"  {dataset} ({file_split}): OK - {detection_str} + {view_str}")
            else:
                print(f"  {dataset}: FILE NOT FOUND")

def verify_specific_constraint_rules():
    """Verify that the constraint rules are being applied correctly"""
    
    print(f"\n{'='*80}")
    print("CONSTRAINT RULES VERIFICATION")
    print("="*80)
    
    print("Anatomical Constraint Rules:")
    print("- A4C (0): allows MR(1), TR(3)")
    print("- PSAX (1): allows PR(2), TR(3)") 
    print("- PLAX (2): allows AR(0), MR(1)")
    print()
    
    # Test with a sample of files from each dataset
    datasets = ['regurgitationV1', 'regurgitationV2', 'regurgitationV3']
    
    sample_found = 0
    
    for dataset in datasets:
        train_labels = f'./{dataset}/train/labels'
        if os.path.exists(train_labels):
            files = [f for f in os.listdir(train_labels) if f.endswith('.txt')]
            
            # Check first few files
            for filename in files[:3]:
                file_path = os.path.join(train_labels, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Parse
                    detections = []
                    view_class = None
                    
                    for line in lines:
                        parts = line.split()
                        
                        if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                            for j, val in enumerate(parts):
                                if val == '1':
                                    view_class = j
                                    break
                        elif len(parts) >= 5:
                            try:
                                detection_class = int(parts[0])
                                detections.append(detection_class)
                            except:
                                continue
                    
                    # Check constraints
                    if view_class is not None and detections:
                        allowed = {
                            0: [1, 3],  # A4C
                            1: [2, 3],  # PSAX  
                            2: [0, 1],  # PLAX
                        }.get(view_class, [])
                        
                        violations = [d for d in detections if d not in allowed]
                        
                        if violations:
                            print(f"[VIOLATION] {dataset}/{filename}: {detections} + view {view_class}")
                        
                        sample_found += 1
                        
                        if sample_found >= 9:  # Stop after checking enough samples
                            break
                            
                except:
                    continue
            
            if sample_found >= 9:
                break
    
    print("Sample verification complete - no violations reported above means all are OK")

if __name__ == "__main__":
    # Check specific previously violated files
    check_specific_violations()
    
    # Verify constraint rules are working
    verify_specific_constraint_rules()
