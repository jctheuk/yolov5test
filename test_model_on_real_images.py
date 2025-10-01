#!/usr/bin/env python3
"""
Test model predictions on real dataset images
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_model_on_real_images():
    """Test model predictions on real dataset images"""
    print("TESTING MODEL ON REAL DATASET IMAGES")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        print("OK: Model loaded successfully")
        
        # Create dataloader with shuffle to get different samples
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
        
        print("Testing model on real images...")
        
        class_names = ['A4C', 'PSAX', 'PLAX']
        prediction_counts = {0: 0, 1: 0, 2: 0}  # A4C, PSAX, PLAX
        correct_predictions = 0
        total_predictions = 0
        
        # Test multiple batches
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Test first 5 batches
                break
                
            images, targets, paths, shapes, classification_labels = batch
            
            print(f"\nBatch {batch_idx}:")
            
            with torch.no_grad():
                model_output = model(images)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                detection_output, classification_output = model_output
                
                probs = torch.softmax(classification_output, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
                
                # Count predictions
                for i, (pred_class, prob, true_label) in enumerate(zip(pred_classes, probs, classification_labels)):
                    pred_name = class_names[pred_class.item()]
                    confidence = prob[pred_class].item()
                    
                    # Get true class
                    true_class = true_label.argmax().item()
                    true_name = class_names[true_class]
                    
                    # Count predictions
                    prediction_counts[pred_class.item()] += 1
                    
                    # Check if prediction is correct
                    if pred_class.item() == true_class:
                        correct_predictions += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    total_predictions += 1
                    
                    if i < 3:  # Show first 3 samples per batch
                        print(f"  Sample {i}: Predicted {pred_name} vs True {true_name} (conf: {confidence:.3f}) {status}")
                
                # Show batch summary
                batch_correct = sum(1 for p, t in zip(pred_classes, classification_labels) 
                                  if p.item() == t.argmax().item())
                batch_total = len(pred_classes)
                batch_accuracy = batch_correct / batch_total
                print(f"  Batch accuracy: {batch_accuracy:.3f} ({batch_correct}/{batch_total})")
        
        # Overall summary
        print(f"\nOVERALL SUMMARY:")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall accuracy: {correct_predictions/total_predictions:.3f}")
        
        print(f"\nPrediction distribution:")
        for i, name in enumerate(class_names):
            count = prediction_counts[i]
            percentage = (count / total_predictions) * 100
            print(f"  {name}: {count} predictions ({percentage:.1f}%)")
        
        # Check if model is biased
        max_predictions = max(prediction_counts.values())
        max_class = max(prediction_counts.keys(), key=lambda k: prediction_counts[k])
        
        if max_predictions / total_predictions > 0.8:
            print(f"\nWARNING: Model is heavily biased toward {class_names[max_class]} ({max_predictions/total_predictions:.1%})")
            return False
        else:
            print(f"\nOK: Model predictions are reasonably distributed")
            return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_learning_capability():
    """Test if model can learn from different inputs"""
    print("\nTESTING MODEL LEARNING CAPABILITY")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.train()  # Set to training mode
        
        print("OK: Model set to training mode")
        
        # Test gradient flow
        test_input = torch.randn(2, 3, 416, 416, requires_grad=True)
        test_targets = torch.tensor([0, 1], dtype=torch.long)  # Different classes
        
        print("Testing gradient flow...")
        
        # Forward pass
        model_output = model(test_input)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            
            # Compute loss
            loss = torch.nn.CrossEntropyLoss()(classification_output, test_targets)
            
            print(f"Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check if gradients are computed
            grad_norm = 0
            param_count = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_count += 1
            
            grad_norm = grad_norm ** 0.5
            
            print(f"Gradient norm: {grad_norm:.6f}")
            print(f"Parameters with gradients: {param_count}")
            
            if grad_norm > 0:
                print("OK: Gradients are flowing - model can learn")
                return True
            else:
                print("ERROR: No gradients - model cannot learn")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("MODEL TESTING ON REAL IMAGES")
    print("=" * 60)
    
    # Test 1: Model on real images
    success1 = test_model_on_real_images()
    
    # Test 2: Model learning capability
    success2 = test_model_learning_capability()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1 and success2:
        print("SUCCESS: Model works correctly on real images and can learn")
        print("   The issue might be in training configuration or hyperparameters")
    else:
        print("ERROR: Model has issues with real images or learning")
        print("   This could explain the 40% accuracy problem")

if __name__ == "__main__":
    main()
