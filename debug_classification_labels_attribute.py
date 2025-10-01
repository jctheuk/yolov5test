#!/usr/bin/env python3
"""
Debug the classification_labels attribute in the dataset
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def debug_classification_labels():
    """Debug the classification_labels attribute"""
    print("DEBUG CLASSIFICATION_LABELS ATTRIBUTE")
    print("=" * 50)
    
    # Clear cache first
    cache_files = [
        "regurgitationV1/train/labels.cache",
        "regurgitationV1/train/labels.cache.npy"
    ]
    
    for cache_file in cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_file}")
    
    try:
        from yolov5c.utils.dataloaders import LoadImagesAndLabels
        
        print("Creating dataset...")
        
        # Create dataset directly
        dataset = LoadImagesAndLabels(
            'regurgitationV1/train/images',
            img_size=416,
            batch_size=4,
            augment=False,
            hyp={'cls_task': 0.3},
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            min_items=0,
            prefix='DEBUG: '
        )
        
        print("Dataset created successfully")
        
        # Check classification_labels attribute
        print(f"\nClassification labels analysis:")
        print(f"  Type: {type(dataset.classification_labels)}")
        print(f"  Length: {len(dataset.classification_labels)}")
        
        # Check first 10 entries
        print(f"\nFirst 10 classification labels:")
        for i in range(min(10, len(dataset.classification_labels))):
            cls_label = dataset.classification_labels[i]
            print(f"  Index {i}: {cls_label} (type: {type(cls_label)})")
        
        # Check if all are the same
        unique_labels = set(tuple(label) if isinstance(label, (list, tuple)) else (label,) for label in dataset.classification_labels[:100])
        print(f"\nUnique labels (first 100): {len(unique_labels)}")
        
        if len(unique_labels) == 1:
            print(f"  ERROR: All labels are the same: {list(unique_labels)[0]}")
        else:
            print(f"  OK: Multiple unique labels found")
            for i, label in enumerate(list(unique_labels)[:5]):
                print(f"    Label {i}: {label}")
        
        # Test __getitem__ method directly
        print(f"\nTesting __getitem__ method:")
        sample = dataset[0]
        if len(sample) == 5:
            img, labels, path, shapes, classification_tensor = sample
            print(f"  Sample 0 classification tensor: {classification_tensor}")
            print(f"  Classification tensor type: {type(classification_tensor)}")
            print(f"  Classification tensor shape: {classification_tensor.shape}")
            
            # Check if it matches the classification_labels attribute
            cls_label_attr = dataset.classification_labels[0]
            print(f"  Classification label attribute: {cls_label_attr}")
            print(f"  Classification label attribute type: {type(cls_label_attr)}")
            
            # Convert attribute to tensor for comparison
            if isinstance(cls_label_attr, (list, tuple)):
                attr_tensor = torch.tensor([float(x) for x in cls_label_attr], dtype=torch.float32)
            else:
                attr_tensor = torch.tensor([float(cls_label_attr)], dtype=torch.float32)
            
            print(f"  Converted attribute tensor: {attr_tensor}")
            
            if torch.equal(classification_tensor, attr_tensor):
                print(f"  OK: __getitem__ result matches attribute")
            else:
                print(f"  ERROR: __getitem__ result differs from attribute")
                print(f"    Expected: {attr_tensor}")
                print(f"    Got: {classification_tensor}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_cache_structure():
    """Debug what's actually stored in the cache"""
    print("\nDEBUG CACHE STRUCTURE")
    print("=" * 40)
    
    cache_file = Path("regurgitationV1/train/labels.cache")
    
    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            print(f"Cache loaded successfully")
            print(f"Cache keys: {len(cache)} entries")
            
            # Check structure of first entry
            first_key = list(cache.keys())[0]
            first_value = cache[first_key]
            
            print(f"\nFirst cache entry:")
            print(f"  Key: {Path(first_key).name}")
            print(f"  Value type: {type(first_value)}")
            print(f"  Value length: {len(first_value)}")
            
            if isinstance(first_value, (list, tuple)) and len(first_value) >= 4:
                labels, shape, segments, classification_labels = first_value[:4]
                print(f"  Labels: {labels}")
                print(f"  Shape: {shape}")
                print(f"  Segments: {segments}")
                print(f"  Classification labels: {classification_labels}")
                print(f"  Classification labels type: {type(classification_labels)}")
                
                # Check if this is the correct format
                if isinstance(classification_labels, (list, tuple)) and len(classification_labels) == 3:
                    print(f"  OK: Classification labels format is correct")
                else:
                    print(f"  ERROR: Classification labels format is wrong")
            else:
                print(f"  ERROR: Cache entry format is wrong")
                
        except Exception as e:
            print(f"ERROR loading cache: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Cache file not found")

def main():
    """Main debug function"""
    print("CLASSIFICATION LABELS ATTRIBUTE DEBUG")
    print("=" * 60)
    
    # Test 1: Debug classification_labels attribute
    debug_classification_labels()
    
    # Test 2: Debug cache structure
    debug_cache_structure()

if __name__ == "__main__":
    main()
