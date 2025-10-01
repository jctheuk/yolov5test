#!/usr/bin/env python3
"""
Test what's actually stored in the cache
"""

import sys
import pickle
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_cache_contents():
    """Test what's in the cache file"""
    print("TESTING CACHE CONTENTS")
    print("=" * 40)
    
    cache_file = Path("regurgitationV1/train/labels.cache")
    
    if not cache_file.exists():
        print("ERROR: Cache file not found")
        return
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        print(f"Cache loaded successfully")
        print(f"Cache keys: {len(cache)} entries")
        
        # Check first few entries
        sample_keys = list(cache.keys())[:3]
        print(f"\nSample cache entries:")
        
        for key in sample_keys:
            if key in cache:
                value = cache[key]
                print(f"\n{Path(key).name}:")
                print(f"  Type: {type(value)}")
                print(f"  Length: {len(value)}")
                
                if isinstance(value, (list, tuple)) and len(value) >= 4:
                    labels, shape, segments, classification_labels = value[:4]
                    print(f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                    print(f"  Shape: {shape}")
                    print(f"  Segments: {segments}")
                    print(f"  Classification labels: {classification_labels}")
                    print(f"  Classification labels type: {type(classification_labels)}")
                else:
                    print(f"  Value: {value}")
            else:
                print(f"  Key not found in cache")
        
        # Check if all classification labels are the same
        classification_labels = []
        for key, value in cache.items():
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                classification_labels.append(value[3])
        
        print(f"\nClassification labels analysis:")
        print(f"  Total samples: {len(classification_labels)}")
        
        if classification_labels:
            unique_labels = set(tuple(label) for label in classification_labels)
            print(f"  Unique labels: {len(unique_labels)}")
            
            if len(unique_labels) == 1:
                print(f"  ERROR: All labels are the same: {list(unique_labels)[0]}")
            else:
                print(f"  OK: Multiple unique labels found")
                for i, label in enumerate(list(unique_labels)[:5]):
                    print(f"    Label {i}: {label}")
        
    except Exception as e:
        print(f"ERROR loading cache: {e}")
        import traceback
        traceback.print_exc()

def test_specific_file_in_cache():
    """Test a specific file that we know has classification labels"""
    print("\nTESTING SPECIFIC FILE IN CACHE")
    print("=" * 40)
    
    cache_file = Path("regurgitationV1/train/labels.cache")
    
    if not cache_file.exists():
        print("ERROR: Cache file not found")
        return
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        # Test the specific file we know has classification labels
        test_key = None
        for key in cache.keys():
            if "a2hiwqVqZ2o=-unnamed_1_1.mp4-0" in str(key):
                test_key = key
                break
        
        if test_key:
            print(f"Found test file: {Path(test_key).name}")
            value = cache[test_key]
            
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                labels, shape, segments, classification_labels = value[:4]
                print(f"  Labels: {labels}")
                print(f"  Classification labels: {classification_labels}")
                
                # This should be [0.0, 1.0, 0.0] for PSAX
                if classification_labels == [0.0, 1.0, 0.0]:
                    print("  OK: Classification label is correct (PSAX)")
                else:
                    print(f"  ERROR: Expected [0.0, 1.0, 0.0], got {classification_labels}")
            else:
                print(f"  ERROR: Invalid cache entry format")
        else:
            print("ERROR: Test file not found in cache")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("CACHE CONTENTS ANALYSIS")
    print("=" * 50)
    
    test_cache_contents()
    test_specific_file_in_cache()

if __name__ == "__main__":
    main()
