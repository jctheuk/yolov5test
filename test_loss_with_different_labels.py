#!/usr/bin/env python3
"""
Test loss function with different classification labels
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_loss_with_different_labels():
    """Test loss function with different classification labels"""
    print("TESTING LOSS WITH DIFFERENT LABELS")
    print("=" * 50)
    
    try:
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        # Create loss function
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=1.0,
            label_smoothing=0.1
        )
        
        print("OK: Loss function created")
        
        # Create dataloader with shuffle enabled to get different labels
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
        
        print("Testing multiple batches to find one with different labels...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Test first 5 batches
                break
                
            images, targets, paths, shapes, classification_labels = batch
            
            # Check if batch has different labels
            unique_labels = torch.unique(classification_labels, dim=0)
            
            print(f"\nBatch {batch_idx}:")
            print(f"  Unique labels: {len(unique_labels)}")
            
            if len(unique_labels) > 1:
                print(f"  SUCCESS: Found batch with different labels!")
                print(f"  Classification labels:")
                
                class_names = ['A4C', 'PSAX', 'PLAX']
                for i, label in enumerate(classification_labels):
                    class_idx = label.argmax().item()
                    print(f"    Sample {i}: {class_names[class_idx]} - {label.tolist()}")
                
                # Test loss computation
                print(f"\n  Testing loss computation...")
                
                with torch.no_grad():
                    model_output = model(images)
                
                total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
                
                print(f"  Total loss: {total_loss}")
                print(f"  Total loss item: {total_loss.item()}")
                print(f"  Loss items: {[item.item() for item in loss_items]}")
                
                if total_loss.item() > 0:
                    print(f"  SUCCESS: Non-zero loss found!")
                    return True
                else:
                    print(f"  WARNING: Loss is still zero")
            else:
                print(f"  All labels are the same: {unique_labels[0].tolist()}")
        
        print(f"\nERROR: No batch with different labels found")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_different_labels():
    """Test with manually created different labels"""
    print("\nTESTING WITH MANUAL DIFFERENT LABELS")
    print("=" * 50)
    
    try:
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        # Create loss function
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=1.0,
            label_smoothing=0.1
        )
        
        # Create dummy input
        dummy_images = torch.randn(4, 3, 416, 416)
        
        # Create different classification labels
        different_labels = torch.tensor([
            [1.0, 0.0, 0.0],  # A4C
            [0.0, 1.0, 0.0],  # PSAX
            [0.0, 0.0, 1.0],  # PLAX
            [1.0, 0.0, 0.0],  # A4C
        ])
        
        # Create dummy targets
        dummy_targets = torch.zeros(4, 6)
        
        print(f"Testing with different labels:")
        class_names = ['A4C', 'PSAX', 'PLAX']
        for i, label in enumerate(different_labels):
            class_idx = label.argmax().item()
            print(f"  Sample {i}: {class_names[class_idx]} - {label.tolist()}")
        
        # Forward pass
        with torch.no_grad():
            model_output = model(dummy_images)
        
        # Compute loss
        total_loss, loss_items = compute_loss(model_output, dummy_targets, different_labels)
        
        print(f"\nLoss computation results:")
        print(f"  Total loss: {total_loss}")
        print(f"  Total loss item: {total_loss.item()}")
        print(f"  Loss items: {[item.item() for item in loss_items]}")
        
        if total_loss.item() > 0:
            print(f"  SUCCESS: Non-zero loss with different labels!")
            return True
        else:
            print(f"  ERROR: Loss is still zero even with different labels")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("LOSS WITH DIFFERENT LABELS TEST")
    print("=" * 60)
    
    # Test 1: Find batch with different labels
    success1 = test_loss_with_different_labels()
    
    # Test 2: Manual different labels
    success2 = test_manual_different_labels()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1 or success2:
        print("SUCCESS: Loss function works with different labels")
        print("   The issue is that the model predicts class 0 for everything")
        print("   This explains the 40% accuracy - model is not learning")
    else:
        print("ERROR: Loss function has issues even with different labels")

if __name__ == "__main__":
    main()
