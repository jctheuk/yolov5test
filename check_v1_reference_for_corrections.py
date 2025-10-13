#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check V1 Reference for Corrections
Compare V1 (correct) with V2-V5 (need corrections) to identify proper classification labels
"""

import os
import json
from collections import defaultdict

def parse_label_file(label_path):
    """Parse label file and return detections and view class"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        detections = []
        view_class = None
        
        for line in lines:
            parts = line.split()
            
            # Classification line (3 elements, all 0 or 1)
            if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                for i, val in enumerate(parts):
                    if val == '1':
                        view_class = i
                        break
                        
            # Detection line (5+ elements)
            elif len(parts) >= 5:
                try:
                    detection_class = int(parts[0])
                    detections.append(detection_class)
                except ValueError:
                    continue
        
        return detections, view_class
        
    except Exception as e:
        return None, None

def get_correct_classification_for_detections(detections):
    """
    Determine correct view classification based on detection classes
    Based on anatomical constraints:
    - If detections contain MR(1) or TR(3): should be A4C(0) or PSAX(1) 
    - If detections contain PR(2) or TR(3): should be PSAX(1)
    - If detections contain AR(0) or MR(1): should be PLAX(2)
    """
    
    # Detection class names for reference
    detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
    view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
    
    # Valid combinations according to anatomical constraints
    valid_combinations = {
        0: [1, 3],  # A4C: allows MR, TR
        1: [2, 3],  # PSAX: allows PR, TR  
        2: [0, 1],  # PLAX: allows AR, MR
    }
    
    detection_set = set(detections)
    
    # Find which view class allows all detections
    for view_class, allowed_detections in valid_combinations.items():
        if detection_set.issubset(set(allowed_detections)):
            return view_class, view_names[view_class]
    
    # If no single view allows all detections, find the best match
    best_matches = []
    for view_class, allowed_detections in valid_combinations.items():
        allowed_set = set(allowed_detections)
        matches = len(detection_set.intersection(allowed_set))
        if matches > 0:
            best_matches.append((view_class, matches, view_names[view_class]))
    
    if best_matches:
        # Sort by number of matches, return the best one
        best_matches.sort(key=lambda x: x[1], reverse=True)
        return best_matches[0][0], best_matches[0][2]
    
    return None, None

def analyze_violation_files():
    """Analyze violation files and suggest corrections"""
    
    print("="*80)
    print("V1 REFERENCE ANALYSIS FOR V2-V5 CORRECTIONS")
    print("="*80)
    
    # Load violation files from V2
    violation_file = './violation_analysis/regurgitationV2_constraint_violation_filenames.txt'
    
    if not os.path.exists(violation_file):
        print(f"Error: {violation_file} not found")
        return
    
    with open(violation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    violation_files = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            violation_files.append(line)
    
    print(f"Found {len(violation_files)} violation files to analyze")
    
    # Analysis results
    corrections = []
    v1_missing = []
    
    view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
    detection_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
    
    print(f"\nAnalyzing each violation file:")
    print("-" * 50)
    
    for i, filename in enumerate(violation_files[:10]):  # Analyze first 10 as examples
        print(f"\n{i+1}. {filename}")
        
        # Check V1 (reference)
        v1_splits = ['train', 'valid', 'test']
        v1_found = False
        v1_detections = None
        v1_view = None
        v1_split = None
        
        for split in v1_splits:
            v1_path = f'./regurgitationV1/{split}/labels/{filename}'
            if os.path.exists(v1_path):
                v1_detections, v1_view = parse_label_file(v1_path)
                v1_split = split
                v1_found = True
                break
        
        if not v1_found:
            print(f"   [MISSING] Not found in V1 - file may have been removed during cleaning")
            v1_missing.append(filename)
            continue
        
        # Check V2 (needs correction)
        v2_splits = ['train', 'valid', 'test']
        v2_found = False
        v2_detections = None
        v2_view = None
        v2_split = None
        
        for split in v2_splits:
            v2_path = f'./regurgitationV2/{split}/labels/{filename}'
            if os.path.exists(v2_path):
                v2_detections, v2_view = parse_label_file(v2_path)
                v2_split = split
                v2_found = True
                break
        
        if not v2_found:
            print(f"   [ERROR] Not found in V2")
            continue
        
        # Compare and analyze
        print(f"   V1 ({v1_split}): detections={v1_detections} -> view={v1_view}({view_names.get(v1_view, 'None')})")
        print(f"   V2 ({v2_split}): detections={v2_detections} -> view={v2_view}({view_names.get(v2_view, 'None')})")
        
        # Determine correct classification based on detections
        correct_view, correct_view_name = get_correct_classification_for_detections(v2_detections)
        
        if correct_view is not None:
            print(f"   [CORRECT] Classification should be: {correct_view}({correct_view_name})")
            
            correction = {
                'filename': filename,
                'v1_split': v1_split,
                'v2_split': v2_split,
                'detections': v2_detections,
                'detection_names': [detection_names[d] for d in v2_detections],
                'current_view': v2_view,
                'current_view_name': view_names.get(v2_view, 'Unknown'),
                'correct_view': correct_view,
                'correct_view_name': correct_view_name,
                'v1_reference_view': v1_view,
                'v1_reference_view_name': view_names.get(v1_view, 'Unknown')
            }
            corrections.append(correction)
            
        else:
            print(f"   [WARNING] Cannot determine correct classification for detections: {v2_detections}")
    
    # Summary
    print(f"\n" + "="*80)
    print("CORRECTION SUMMARY")
    print("="*80)
    
    print(f"\nFiles missing in V1: {len(v1_missing)}")
    if v1_missing:
        print("These files were likely removed during V1 cleaning:")
        for filename in v1_missing[:5]:
            print(f"  - {filename}")
        if len(v1_missing) > 5:
            print(f"  ... and {len(v1_missing) - 5} more")
    
    print(f"\nCorrections needed: {len(corrections)}")
    
    if corrections:
        print("\nCorrection examples:")
        for correction in corrections[:5]:
            print(f"  {correction['filename']}:")
            print(f"    Detections: {correction['detection_names']}")
            print(f"    Current: {correction['current_view_name']} -> Correct: {correction['correct_view_name']}")
            if correction['v1_reference_view'] is not None:
                print(f"    V1 reference: {correction['v1_reference_view_name']}")
    
    return corrections, v1_missing

if __name__ == "__main__":
    corrections, missing = analyze_violation_files()
