#!/usr/bin/env python3
"""
Test dataloader without cache to see if classification labels work
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_dataloader_no_cache():
    """Test dataloader with cache disabled"""
    print("TESTING DATALOADER WITHOUT CACHE")
    print("=" * 50)
    
    # Clear any existing cache
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
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Creating dataloader with cache disabled...")
        
        # Create dataloader with cache disabled
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=4,
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,  # Disable cache
            rect=False,
            rank=-1,
            workers=0,
            prefix='',
            shuffle=False
        )
        
        print("OK: Dataloader created successfully")
        
        # Test first batch
        batch = next(iter(train_loader))
        images, targets, paths, shapes, classification_labels = batch
        
        print(f"Batch analysis:")
        print(f"  Images shape: {images.shape}")
        print(f"  Classification labels shape: {classification_labels.shape}")
        print(f"  Classification labels: {classification_labels}")
        
        # Check if labels are different
        unique_labels = torch.unique(classification_labels, dim=0)
        print(f"  Unique classification labels: {len(unique_labels)}")
        
        if len(unique_labels) > 1:
            print("SUCCESS: Multiple unique classification labels found!")
            print("   Classification labels are working correctly")
            
            # Show class distribution in batch
            class_names = ['A4C', 'PSAX', 'PLAX']
            for i, label in enumerate(classification_labels):
                class_idx = label.argmax().item()
                confidence = label.max().item()
                print(f"    Sample {i}: {class_names[class_idx]} (confidence: {confidence:.3f})")
            
            return True
        else:
            print("ERROR: All samples have same classification label")
            print("   The issue persists even without cache")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_directly():
    """Test dataset directly without dataloader"""
    print("\nTESTING DATASET DIRECTLY")
    print("=" * 40)
    
    try:
        from yolov5c.utils.dataloaders import LoadImagesAndLabels
        
        print("Creating dataset directly...")
        
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
            prefix=''
        )
        
        print("OK: Dataset created successfully")
        
        # Test first few samples
        print("Testing first 5 samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if len(sample) == 5:
                img, labels, path, shapes, classification_label = sample
                print(f"  Sample {i}:")
                print(f"    Image shape: {img.shape}")
                print(f"    Labels shape: {labels.shape}")
                print(f"    Classification label: {classification_label}")
                print(f"    Classification label type: {type(classification_label)}")
            else:
                print(f"  Sample {i}: Unexpected format, length {len(sample)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("DATALOADER NO-CACHE TEST")
    print("=" * 60)
    
    # Test 1: Dataloader without cache
    success1 = test_dataloader_no_cache()
    
    # Test 2: Dataset directly
    success2 = test_dataset_directly()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1:
        print("SUCCESS: Dataloader works correctly without cache")
        print("   The issue was with the cache system")
        print("   Solution: Use cache=False or fix cache loading")
    elif success2:
        print("PARTIAL: Dataset works but dataloader has issues")
        print("   The issue is in the dataloader wrapper")
    else:
        print("ERROR: Issue persists even without cache")
        print("   The problem is deeper in the dataset loading")

if __name__ == "__main__":
    main()
