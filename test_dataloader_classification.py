#!/usr/bin/env python3
"""
Test if dataloader is loading classification labels correctly
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_dataloader_classification():
    """Test if dataloader loads classification labels correctly"""
    print("DATALOADER CLASSIFICATION TEST")
    print("=" * 50)
    
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Creating dataloader...")
        
        # Create dataloader with minimal settings
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=2,  # Small batch for testing
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,  # Disable cache to avoid issues
            rect=False,
            rank=-1,
            workers=0,  # No multiprocessing
            prefix='',
            shuffle=False
        )
        
        print("OK: Dataloader created successfully")
        
        # Test first batch
        print("\nTesting first batch...")
        batch = next(iter(train_loader))
        
        if len(batch) == 5:
            images, targets, paths, shapes, classification_labels = batch
            print(f"OK: Batch unpacked correctly")
            print(f"   Images shape: {images.shape}")
            print(f"   Targets shape: {targets.shape}")
            print(f"   Classification labels shape: {classification_labels.shape}")
            print(f"   Classification labels: {classification_labels}")
            
            # Check if classification labels are different (not all default)
            unique_labels = torch.unique(classification_labels, dim=0)
            print(f"   Unique classification labels: {len(unique_labels)}")
            
            if len(unique_labels) == 1:
                print("ERROR: All samples have same classification label!")
                print("   This means classification labels are defaulting to class 0")
                print("   The dataloader is NOT loading labels from files correctly")
                return False
            else:
                print("OK: Multiple unique classification labels found")
                
                # Show class distribution in batch
                class_names = ['A4C', 'PSAX', 'PLAX']
                for i, label in enumerate(classification_labels):
                    class_idx = label.argmax().item()
                    confidence = label.max().item()
                    print(f"   Sample {i}: {class_names[class_idx]} (confidence: {confidence:.3f})")
                
                return True
        else:
            print(f"ERROR: Expected 5 batch elements, got {len(batch)}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_issue():
    """Test if cache is causing the problem"""
    print("\nCACHE TEST")
    print("=" * 30)
    
    cache_file = Path("regurgitationV1/train/labels.cache")
    cache_npy_file = Path("regurgitationV1/train/labels.cache.npy")
    
    if cache_file.exists():
        print(f"Found cache file: {cache_file}")
        
        # Try to read cache
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            print(f"Cache loaded: {len(cache)} entries")
            
            # Check first few entries
            sample_keys = list(cache.keys())[:2]
            for key in sample_keys:
                value = cache[key]
                if isinstance(value, tuple) and len(value) >= 4:
                    labels, shape, segments, classification_labels = value[:4]
                    print(f"  {Path(key).name}: classification_labels = {classification_labels}")
                else:
                    print(f"  {Path(key).name}: unexpected format")
                    
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    if cache_npy_file.exists():
        print(f"Found cache npy file: {cache_npy_file}")
    
    # Check if we should clear cache
    if cache_file.exists() or cache_npy_file.exists():
        print("\nRECOMMENDATION: Clear cache files and retry")
        print("   The cache might contain old/wrong classification labels")

def main():
    """Run dataloader test"""
    print("TESTING DATALOADER CLASSIFICATION LOADING")
    print("=" * 60)
    
    # Test 1: Check if dataloader loads classification labels
    success = test_dataloader_classification()
    
    # Test 2: Check cache issues
    test_cache_issue()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success:
        print("OK: Dataloader is loading classification labels correctly")
        print("   The issue is likely in the model architecture or loss function")
        print("\nNEXT STEPS:")
        print("   1. Check if model is receiving classification labels")
        print("   2. Verify loss function is using classification labels")
        print("   3. Check model output format")
    else:
        print("ERROR: Dataloader is NOT loading classification labels correctly")
        print("   This is the root cause of poor classification performance")
        print("\nSOLUTIONS:")
        print("   1. Clear cache files: labels.cache, labels.cache.npy")
        print("   2. Check dataloader code in yolov5c/utils/dataloaders.py")
        print("   3. Verify label file parsing in verify_image_label()")

if __name__ == "__main__":
    main()
