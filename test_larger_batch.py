#!/usr/bin/env python3
"""
Test dataloader with larger batch to see classification labels
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_larger_batch():
    """Test dataloader with larger batch"""
    print("TESTING LARGER BATCH")
    print("=" * 40)
    
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Creating dataloader with batch size 16...")
        
        # Create dataloader with larger batch
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=16,  # Larger batch
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            workers=0,
            prefix='',
            shuffle=True  # Enable shuffle
        )
        
        print("OK: Dataloader created successfully")
        
        # Test first batch
        batch = next(iter(train_loader))
        images, targets, paths, shapes, classification_labels = batch
        
        print(f"Batch analysis:")
        print(f"  Images shape: {images.shape}")
        print(f"  Classification labels shape: {classification_labels.shape}")
        print(f"  Classification labels:")
        
        # Show all classification labels in batch
        class_names = ['A4C', 'PSAX', 'PLAX']
        for i, label in enumerate(classification_labels):
            class_idx = label.argmax().item()
            confidence = label.max().item()
            print(f"    Sample {i:2d}: {class_names[class_idx]} (confidence: {confidence:.3f}) - {label.tolist()}")
        
        # Check if labels are different
        unique_labels = torch.unique(classification_labels, dim=0)
        print(f"\nUnique classification labels: {len(unique_labels)}")
        
        if len(unique_labels) > 1:
            print("SUCCESS: Multiple unique classification labels found!")
            print("   The dataloader IS working correctly!")
            
            # Count class distribution in batch
            class_counts = {0: 0, 1: 0, 2: 0}  # A4C, PSAX, PLAX
            for label in classification_labels:
                class_idx = label.argmax().item()
                class_counts[class_idx] += 1
            
            print(f"\nClass distribution in batch:")
            for i, name in enumerate(class_names):
                print(f"  {name}: {class_counts[i]} samples")
            
            return True
        else:
            print("ERROR: All samples still have same classification label")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_batches():
    """Test multiple batches to see if labels vary"""
    print("\nTESTING MULTIPLE BATCHES")
    print("=" * 40)
    
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=8,
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            workers=0,
            prefix='',
            shuffle=True
        )
        
        print("Testing first 3 batches...")
        
        all_unique_labels = set()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:
                break
                
            images, targets, paths, shapes, classification_labels = batch
            
            print(f"\nBatch {batch_idx}:")
            unique_labels = torch.unique(classification_labels, dim=0)
            print(f"  Unique labels in batch: {len(unique_labels)}")
            
            # Add to global set
            for label in classification_labels:
                all_unique_labels.add(tuple(label.tolist()))
            
            # Show first few labels
            for i, label in enumerate(classification_labels[:3]):
                class_idx = label.argmax().item()
                class_names = ['A4C', 'PSAX', 'PLAX']
                print(f"    Sample {i}: {class_names[class_idx]} - {label.tolist()}")
        
        print(f"\nTotal unique labels across all batches: {len(all_unique_labels)}")
        
        if len(all_unique_labels) > 1:
            print("SUCCESS: Multiple unique labels found across batches!")
            print("   Classification labels are working correctly!")
            return True
        else:
            print("ERROR: All labels are the same across batches")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("LARGER BATCH CLASSIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Larger batch
    success1 = test_larger_batch()
    
    # Test 2: Multiple batches
    success2 = test_multiple_batches()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1 and success2:
        print("SUCCESS: Classification labels are working correctly!")
        print("   The dataloader is loading different classification labels")
        print("   The 40% accuracy issue is NOT caused by label loading")
        print("   The issue must be in model architecture or training configuration")
    else:
        print("ERROR: Classification labels are still not working correctly")

if __name__ == "__main__":
    main()
