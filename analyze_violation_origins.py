#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Violation File Origins
Check if violations across V2-V5 datasets share same names or originate from same videos
"""

import os
import json
import re
from collections import defaultdict, Counter

def extract_video_id(filename):
    """
    Extract video ID from filename
    Examples:
    - 'bWplwqlsaMKZ-unnamed_1_1.mp4-1.txt' -> 'bWplwqlsaMKZ-unnamed_1_1.mp4'
    - 'aGdjwqtqa8Kb-unnamed_1_5.mp4-17.txt' -> 'aGdjwqtqa8Kb-unnamed_1_5.mp4'
    """
    # Remove .txt extension and frame number
    if filename.endswith('.txt'):
        filename = filename[:-4]
    
    # Pattern: video_id-frame_number
    # The video ID usually ends with .mp4
    parts = filename.split('-')
    if len(parts) >= 2:
        # Rejoin all parts except the last one (frame number)
        video_id = '-'.join(parts[:-1])
        if not video_id.endswith('.mp4'):
            video_id += '.mp4'
        return video_id
    
    return filename

def extract_base_video_name(filename):
    """
    Extract the base video name (without _1, _2, etc. variations)
    Examples:
    - 'bWplwqlsaMKZ-unnamed_1_1.mp4' -> 'bWplwqlsaMKZ-unnamed'
    - 'aGdjwqtqa8Kb-unnamed_1_5.mp4' -> 'aGdjwqtqa8Kb-unnamed'
    """
    video_id = extract_video_id(filename)
    # Remove .mp4 extension
    if video_id.endswith('.mp4'):
        video_id = video_id[:-4]
    
    # Pattern: base_name_part_segment
    # Remove _part_segment pattern (like _1_1, _2_1, etc.)
    # Use regex to find the pattern _\d+_\d+$ at the end
    match = re.search(r'(.+)_\d+_\d+$', video_id)
    if match:
        return match.group(1)
    
    return video_id

def load_violation_files():
    """Load violation files from all V2-V5 datasets"""
    datasets = ['regurgitationV2', 'regurgitationV3', 'regurgitationV4', 'regurgitationV5']
    violation_data = {}
    
    for dataset in datasets:
        violation_file = f'./violation_analysis/{dataset}_constraint_violation_filenames.txt'
        
        if os.path.exists(violation_file):
            with open(violation_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header lines (starting with #)
            violation_files = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    violation_files.append(line)
            
            violation_data[dataset] = violation_files
            print(f"Loaded {len(violation_files)} violation files from {dataset}")
        else:
            print(f"Warning: {violation_file} not found")
    
    return violation_data

def analyze_violation_patterns(violation_data):
    """Analyze patterns in violation files across datasets"""
    
    print("\n" + "="*80)
    print("VIOLATION FILE PATTERN ANALYSIS")
    print("="*80)
    
    # Check if all datasets have the same violation files
    print("\n1. EXACT FILENAME COMPARISON")
    print("-" * 40)
    
    datasets = list(violation_data.keys())
    if len(datasets) > 1:
        base_dataset = datasets[0]
        base_files = set(violation_data[base_dataset])
        
        all_same = True
        for dataset in datasets[1:]:
            current_files = set(violation_data[dataset])
            if base_files != current_files:
                all_same = False
                print(f"  {dataset} differs from {base_dataset}")
                
                only_in_base = base_files - current_files
                only_in_current = current_files - base_files
                
                if only_in_base:
                    print(f"    Only in {base_dataset}: {list(only_in_base)[:3]}...")
                if only_in_current:
                    print(f"    Only in {dataset}: {list(only_in_current)[:3]}...")
            else:
                print(f"  {dataset} has identical files to {base_dataset}")
        
        if all_same:
            print("  [SUCCESS] ALL DATASETS HAVE IDENTICAL VIOLATION FILES!")
        else:
            print("  [ERROR] Datasets have different violation files")
    
    # Analyze video origins
    print("\n2. VIDEO ORIGIN ANALYSIS")
    print("-" * 40)
    
    all_video_ids = defaultdict(list)  # video_id -> [dataset1, dataset2, ...]
    all_base_videos = defaultdict(list)  # base_video -> [dataset1, dataset2, ...]
    
    for dataset, files in violation_data.items():
        video_ids = set()
        base_videos = set()
        
        for filename in files:
            video_id = extract_video_id(filename)
            base_video = extract_base_video_name(filename)
            
            video_ids.add(video_id)
            base_videos.add(base_video)
            
            all_video_ids[video_id].append(dataset)
            all_base_videos[base_video].append(dataset)
        
        print(f"\n  {dataset}:")
        print(f"    Unique video IDs: {len(video_ids)}")
        print(f"    Unique base videos: {len(base_videos)}")
        print(f"    Sample video IDs: {list(video_ids)[:3]}...")
        print(f"    Sample base videos: {list(base_videos)[:3]}...")
    
    # Show videos that appear in multiple datasets
    print("\n3. CROSS-DATASET VIDEO ANALYSIS")
    print("-" * 40)
    
    print("\nVideo IDs appearing in multiple datasets:")
    cross_dataset_videos = {vid: datasets for vid, datasets in all_video_ids.items() 
                           if len(datasets) > 1}
    
    if cross_dataset_videos:
        for video_id, datasets in sorted(cross_dataset_videos.items()):
            print(f"  {video_id}: appears in {len(datasets)} datasets ({', '.join(datasets)})")
    else:
        print("  No videos appear in multiple datasets")
    
    print("\nBase videos appearing in multiple datasets:")
    cross_dataset_base_videos = {vid: datasets for vid, datasets in all_base_videos.items() 
                                if len(datasets) > 1}
    
    if cross_dataset_base_videos:
        for base_video, datasets in sorted(cross_dataset_base_videos.items()):
            print(f"  {base_video}: appears in {len(datasets)} datasets ({', '.join(datasets)})")
    else:
        print("  No base videos appear in multiple datasets")

def analyze_detailed_violations():
    """Analyze detailed violation information from JSON file"""
    
    json_file = './violation_analysis/constraint_violations_v2_v5_analysis.json'
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("DETAILED VIOLATION ANALYSIS")
    print("="*80)
    
    # Analyze violation types and their origins
    violation_by_video = defaultdict(list)  # video_id -> [violation_info, ...]
    violation_by_type = defaultdict(list)   # violation_type -> [video_ids, ...]
    
    for dataset_name, dataset_data in data.items():
        if dataset_name == 'summary':
            continue
            
        print(f"\n--- {dataset_name} ---")
        
        for violation in dataset_data.get('violation_details', []):
            filename = violation['file']
            violation_type = violation['violation_type']
            view_name = violation['view_name']
            detection_name = violation['detection_name']
            
            video_id = extract_video_id(filename)
            base_video = extract_base_video_name(filename)
            
            violation_info = {
                'dataset': dataset_name,
                'filename': filename,
                'video_id': video_id,
                'base_video': base_video,
                'violation_type': violation_type,
                'view_name': view_name,
                'detection_name': detection_name
            }
            
            violation_by_video[video_id].append(violation_info)
            violation_by_type[violation_type].append(video_id)
    
    # Show videos with multiple violations
    print(f"\nVideos with multiple violations:")
    multi_violation_videos = {vid: violations for vid, violations in violation_by_video.items() 
                             if len(violations) > 1}
    
    for video_id, violations in sorted(multi_violation_videos.items()):
        print(f"  {video_id}: {len(violations)} violations")
        for violation in violations:
            print(f"    - {violation['violation_type']} in {violation['dataset']}")
    
    # Show violation type patterns
    print(f"\nViolation type patterns:")
    for violation_type, video_ids in sorted(violation_by_type.items()):
        unique_videos = set(video_ids)
        print(f"  {violation_type}: {len(video_ids)} occurrences from {len(unique_videos)} unique videos")
        
        # Count occurrences per video
        video_counts = Counter(video_ids)
        frequent_videos = {vid: count for vid, count in video_counts.items() if count > 1}
        
        if frequent_videos:
            print(f"    Videos with multiple {violation_type} violations:")
            for vid, count in sorted(frequent_videos.items()):
                print(f"      {vid}: {count} times")

def main():
    """Main analysis function"""
    
    print("VIOLATION ORIGIN ANALYSIS")
    print("Checking if violations across V2-V5 datasets share same origins")
    print("="*80)
    
    # Load violation files
    violation_data = load_violation_files()
    
    if not violation_data:
        print("No violation data found. Please ensure violation analysis has been run.")
        return
    
    # Analyze patterns
    analyze_violation_patterns(violation_data)
    
    # Analyze detailed violations
    analyze_detailed_violations()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
