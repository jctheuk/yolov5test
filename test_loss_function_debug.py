#!/usr/bin/env python3
"""
Debug the loss function to find why total_loss is -0.0
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def debug_loss_function():
    """Debug the ClassificationTaskLoss function"""
    print("DEBUG LOSS FUNCTION")
    print("=" * 40)
    
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
        
        # Get a batch
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=4,
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            workers=0,
            prefix='',
            shuffle=False
        )
        
        batch = next(iter(train_loader))
        images, targets, paths, shapes, classification_labels = batch
        
        print(f"Batch info:")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Classification labels: {classification_labels}")
        
        # Forward pass
        print("\nForward pass...")
        with torch.no_grad():
            model_output = model(images)
        
        print(f"Model output type: {type(model_output)}")
        print(f"Model output length: {len(model_output)}")
        
        # Parse outputs
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            print(f"Detection output: {detection_output}")
            print(f"Classification output: {classification_output}")
            print(f"Classification output shape: {classification_output.shape}")
        else:
            print("ERROR: Model output is not a tuple of length 2")
            return False
        
        # Test loss computation step by step
        print("\nTesting loss computation step by step...")
        
        # Test 1: Check if classification_output is not None
        print(f"1. Classification output is not None: {classification_output is not None}")
        
        # Test 2: Check if cls_targets is not None
        print(f"2. Classification labels is not None: {classification_labels is not None}")
        
        # Test 3: Check classification_labels processing
        cls_targets = classification_labels
        print(f"3. Classification labels shape: {cls_targets.shape}")
        print(f"   Classification labels dtype: {cls_targets.dtype}")
        print(f"   Classification labels dim: {cls_targets.dim()}")
        
        if cls_targets.dim() > 1 and cls_targets.shape[-1] > 1:
            target_indices = cls_targets.argmax(dim=-1).long()
            print(f"4. One-hot encoded, converted to indices: {target_indices}")
        else:
            target_indices = cls_targets.long()
            print(f"4. Already indices: {target_indices}")
        
        # Test 4: Check if targets are in valid range
        num_classes = classification_output.shape[-1]
        print(f"5. Number of classes: {num_classes}")
        print(f"   Target indices max: {target_indices.max()}")
        print(f"   Target indices min: {target_indices.min()}")
        
        if target_indices.max() >= num_classes:
            target_indices = torch.clamp(target_indices, 0, num_classes - 1)
            print(f"   Clamped target indices: {target_indices}")
        
        # Test 5: Manual cross-entropy calculation
        print("\n6. Manual cross-entropy calculation...")
        
        # Convert to float for calculation
        classification_output_f = classification_output.float()
        target_indices_f = target_indices.long()
        
        print(f"   Classification output (float): {classification_output_f}")
        print(f"   Target indices (long): {target_indices_f}")
        
        # Manual cross-entropy
        log_probs = torch.log_softmax(classification_output_f, dim=-1)
        print(f"   Log probabilities: {log_probs}")
        
        # Get log probabilities for target classes
        nll_loss = -log_probs.gather(1, target_indices_f.unsqueeze(1)).squeeze(1)
        print(f"   NLL loss: {nll_loss}")
        
        manual_loss = nll_loss.mean()
        print(f"   Manual cross-entropy loss: {manual_loss}")
        
        # Test 6: Call the actual loss function
        print("\n7. Calling actual loss function...")
        
        total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
        
        print(f"   Total loss: {total_loss}")
        print(f"   Total loss type: {type(total_loss)}")
        print(f"   Total loss is None: {total_loss is None}")
        
        if total_loss is not None:
            print(f"   Total loss item: {total_loss.item()}")
            print(f"   Total loss is NaN: {torch.isnan(total_loss)}")
            print(f"   Total loss is Inf: {torch.isinf(total_loss)}")
        
        print(f"   Loss items: {loss_items}")
        print(f"   Loss items length: {len(loss_items)}")
        
        for i, item in enumerate(loss_items):
            print(f"     Loss item {i}: {item}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("LOSS FUNCTION DEBUG")
    print("=" * 50)
    
    debug_loss_function()

if __name__ == "__main__":
    main()
