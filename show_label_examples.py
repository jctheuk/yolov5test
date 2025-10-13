#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show Label Examples
Display actual label file contents from V1 vs V2-V5 to show differences
"""

import os

def read_label_file(file_path):
    """Read and return label file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return None

def parse_label_content(content):
    """Parse label content and explain it"""
    if not content:
        return None
    
    lines = content.strip().split('\n')
    
    # Detection class names
    detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
    view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
    
    detections = []
    classification = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        
        # Check if classification line (3 elements, all 0 or 1)
        if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
            for i, val in enumerate(parts):
                if val == '1':
                    classification = i
                    break
        
        # Check if detection line (5+ elements)
        elif len(parts) >= 5:
            try:
                detection_class = int(parts[0])
                detections.append(detection_class)
            except:
                continue
    
    return {
        'detections': detections,
        'detection_names': [detection_names.get(d, f'Unknown({d})') for d in detections],
        'classification': classification,
        'classification_name': view_names.get(classification, f'Unknown({classification})')
    }

def show_examples():
    """Show specific examples of label file differences"""
    
    print("="*80)
    print("LABEL FILE CONTENT COMPARISON: V1 vs V2")
    print("="*80)
    
    # Load some violation files to examine
    violation_files = [
        'ZmNlwq5mZcKcwps=-unnamed_1_1.mp4-0.txt',
        'ZmNmwq5saG5m-unnamed_1_3.mp4-8.txt', 
        'ZmRnwqZla8Kcwp4=-unnamed_2_2.mp4-35.txt',
        'ZmZnwqlqbMKawp0=-unnamed_1_1.mp4-15.txt',
        'bWplwqlsaMKZ-unnamed_1_1.mp4-1.txt'
    ]
    
    for i, filename in enumerate(violation_files, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}: {filename}")
        print('='*60)
        
        # Find file in V1
        v1_content = None
        v1_path = None
        v1_split = None
        
        for split in ['train', 'valid', 'test']:
            path = f'./regurgitationV1/{split}/labels/{filename}'
            if os.path.exists(path):
                v1_content = read_label_file(path)
                v1_path = path
                v1_split = split
                break
        
        # Find file in V2
        v2_content = None
        v2_path = None
        v2_split = None
        
        for split in ['train', 'valid', 'test']:
            path = f'./regurgitationV2/{split}/labels/{filename}'
            if os.path.exists(path):
                v2_content = read_label_file(path)
                v2_path = path
                v2_split = split
                break
        
        print(f"\n--- V1 CONTENT ({v1_split if v1_split else 'NOT FOUND'}) ---")
        if v1_content:
            print("Raw content:")
            for line_num, line in enumerate(v1_content.split('\n'), 1):
                print(f"  {line_num}: {line}")
            
            v1_parsed = parse_label_content(v1_content)
            if v1_parsed:
                print("Parsed:")
                print(f"  Detections: {v1_parsed['detection_names']} (classes: {v1_parsed['detections']})")
                print(f"  View: {v1_parsed['classification_name']} (class: {v1_parsed['classification']})")
        else:
            print("  FILE NOT FOUND - Likely removed during V1 cleaning")
        
        print(f"\n--- V2 CONTENT ({v2_split if v2_split else 'NOT FOUND'}) ---")
        if v2_content:
            print("Raw content:")
            for line_num, line in enumerate(v2_content.split('\n'), 1):
                print(f"  {line_num}: {line}")
            
            v2_parsed = parse_label_content(v2_content)
            if v2_parsed:
                print("Parsed:")
                print(f"  Detections: {v2_parsed['detection_names']} (classes: {v2_parsed['detections']})")
                print(f"  View: {v2_parsed['classification_name']} (class: {v2_parsed['classification']})")
        else:
            print("  FILE NOT FOUND")
        
        # Analysis
        print(f"\n--- ANALYSIS ---")
        if v1_content and v2_content:
            v1_parsed = parse_label_content(v1_content)
            v2_parsed = parse_label_content(v2_content)
            
            if v1_parsed and v2_parsed:
                # Compare detections
                if v1_parsed['detections'] == v2_parsed['detections']:
                    print("  [SAME] DETECTIONS ARE IDENTICAL")
                else:
                    print("  [DIFF] DETECTIONS ARE DIFFERENT")
                    print(f"    V1 detects: {v1_parsed['detection_names']}")
                    print(f"    V2 detects: {v2_parsed['detection_names']}")
                
                # Compare classification
                if v1_parsed['classification'] == v2_parsed['classification']:
                    print("  [SAME] CLASSIFICATIONS ARE IDENTICAL")
                else:
                    print("  [DIFF] CLASSIFICATIONS ARE DIFFERENT")
                    print(f"    V1 view: {v1_parsed['classification_name']}")
                    print(f"    V2 view: {v2_parsed['classification_name']}")
                
                # Show constraint violation analysis
                print("\n  CONSTRAINT ANALYSIS:")
                
                # V1 analysis
                v1_violations = check_violations(v1_parsed['detections'], v1_parsed['classification'])
                print(f"    V1: {v1_violations}")
                
                # V2 analysis  
                v2_violations = check_violations(v2_parsed['detections'], v2_parsed['classification'])
                print(f"    V2: {v2_violations}")
        
        elif v1_content and not v2_content:
            print("  File exists in V1 but not V2")
        elif not v1_content and v2_content:
            print("  File exists in V2 but not V1 (removed during cleaning)")
        else:
            print("  File not found in either dataset")
    
    # Show correction options
    print(f"\n{'='*80}")
    print("CORRECTION OPTIONS")
    print('='*80)
    
    print("""
OPTION 1: Copy V1 completely (Detection + Classification)
- Replace V2-V5 label files with V1 content
- Result: All datasets become identical to V1
- Pros: Guaranteed 0 violations, consistent with cleaned V1
- Cons: Lose V2-V5 detection variations

OPTION 2: Fix classification only (Keep V2-V5 detections)  
- Keep V2-V5 detection results as-is
- Calculate correct view classification based on detections
- Result: V2-V5 keep their detections but get proper classifications
- Pros: Preserve detection diversity, fix constraint violations
- Cons: May still have some edge cases

Examples of what each option would produce:""")

def check_violations(detections, view_class):
    """Check if combination violates anatomical constraints"""
    # Anatomical constraints
    constraints = {
        0: [1, 3],  # A4C: allows MR, TR
        1: [2, 3],  # PSAX: allows PR, TR  
        2: [0, 1],  # PLAX: allows AR, MR
    }
    
    if view_class not in constraints:
        return "Invalid view class"
    
    allowed = constraints[view_class]
    violations = [d for d in detections if d not in allowed]
    
    if violations:
        detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
        view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
        violation_names = [detection_names[v] for v in violations]
        return f"VIOLATION: {view_names[view_class]} cannot have {violation_names}"
    else:
        return "OK - No violations"

if __name__ == "__main__":
    show_examples()
