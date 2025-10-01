#!/usr/bin/env python3
"""
Test model architecture and loss function with classification labels
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_model_with_classification_labels():
    """Test if model processes classification labels correctly"""
    print("TESTING MODEL WITH CLASSIFICATION LABELS")
    print("=" * 50)
    
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return False
    
    try:
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Loading model...")
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        print("OK: Model loaded successfully")
        
        # Create dataloader
        print("Creating dataloader...")
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
        
        # Get a batch
        batch = next(iter(train_loader))
        images, targets, paths, shapes, classification_labels = batch
        
        print(f"Batch info:")
        print(f"  Images shape: {images.shape}")
        print(f"  Classification labels: {classification_labels}")
        
        # Test model forward pass
        print("\nTesting model forward pass...")
        with torch.no_grad():
            model_output = model(images)
        
        print(f"Model output type: {type(model_output)}")
        
        if isinstance(model_output, tuple):
            print(f"Model output length: {len(model_output)}")
            for i, output in enumerate(model_output):
                if hasattr(output, 'shape'):
                    print(f"  Output {i} shape: {output.shape}")
                else:
                    print(f"  Output {i}: {output}")
        
        # Check if model has classification head
        print(f"\nModel architecture:")
        print(f"  Model type: {type(model)}")
        
        if hasattr(model, 'model'):
            print(f"  Model.model type: {type(model.model)}")
            
            # Check for classification head
            if hasattr(model.model, 'classify'):
                print(f"  Classification head found: {model.model.classify}")
            else:
                print(f"  ERROR: No classification head found!")
                
            # Check model structure
            print(f"  Model structure:")
            for name, module in model.model.named_modules():
                if 'classify' in name.lower() or 'cls' in name.lower():
                    print(f"    {name}: {module}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function():
    """Test the loss function with classification labels"""
    print("\nTESTING LOSS FUNCTION")
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
        
        # Test loss computation
        print("\nTesting loss computation...")
        
        # Forward pass
        with torch.no_grad():
            model_output = model(images)
        
        # Compute loss
        total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
        
        print(f"Loss computation results:")
        print(f"  Total loss: {total_loss}")
        print(f"  Loss items: {loss_items}")
        
        # Check if classification loss is being computed
        if hasattr(loss_items, 'cls_task_loss'):
            print(f"  Classification loss: {loss_items.cls_task_loss}")
        else:
            print(f"  ERROR: Classification loss not found in loss items")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classification_output_parsing():
    """Test if classification output is being parsed correctly"""
    print("\nTESTING CLASSIFICATION OUTPUT PARSING")
    print("=" * 50)
    
    try:
        from yolov5c.utils.general import parse_model_output
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 416, 416)
        
        print("Testing model output parsing...")
        
        with torch.no_grad():
            model_output = model(dummy_input)
        
        print(f"Raw model output type: {type(model_output)}")
        
        # Parse output
        detection_output, classification_output = parse_model_output(model_output)
        
        print(f"Parsed outputs:")
        print(f"  Detection output: {detection_output}")
        print(f"  Classification output: {classification_output}")
        
        if classification_output is not None:
            print(f"  Classification output shape: {classification_output.shape}")
            print(f"  Classification output: {classification_output}")
            
            # Check if classification output makes sense
            if classification_output.shape[-1] == 3:  # Should be 3 classes
                print(f"  OK: Classification output has 3 classes")
                
                # Check if outputs are reasonable
                probs = torch.softmax(classification_output, dim=-1)
                print(f"  Classification probabilities: {probs}")
                
                # Check if probabilities sum to 1
                prob_sums = probs.sum(dim=-1)
                print(f"  Probability sums: {prob_sums}")
                
                if torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6):
                    print(f"  OK: Probabilities sum to 1")
                else:
                    print(f"  ERROR: Probabilities don't sum to 1")
                
                return True
            else:
                print(f"  ERROR: Classification output has wrong number of classes")
                return False
        else:
            print(f"  ERROR: No classification output found")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("MODEL CLASSIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Model with classification labels
    success1 = test_model_with_classification_labels()
    
    # Test 2: Loss function
    success2 = test_loss_function()
    
    # Test 3: Classification output parsing
    success3 = test_classification_output_parsing()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1 and success2 and success3:
        print("SUCCESS: Model architecture and loss function appear correct")
        print("   The issue might be in training configuration or hyperparameters")
    else:
        print("ERROR: Issues found in model architecture or loss function")
        print("   This could explain the 40% accuracy problem")

if __name__ == "__main__":
    main()
