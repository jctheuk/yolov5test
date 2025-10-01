#!/usr/bin/env python3
"""
Investigate why the model is biased toward predicting class 0 (A4C)
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def investigate_model_bias():
    """Investigate model prediction bias"""
    print("INVESTIGATING MODEL PREDICTION BIAS")
    print("=" * 50)
    
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return False
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        print("Loading model...")
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        print("OK: Model loaded successfully")
        
        # Test 1: Check model weights initialization
        print("\n1. CHECKING MODEL WEIGHTS")
        print("-" * 30)
        
        # Find classification head weights
        classification_weights = None
        classification_bias = None
        
        for name, param in model.named_parameters():
            if 'classify' in name.lower() or 'cls' in name.lower():
                print(f"  Found classification parameter: {name}")
                print(f"    Shape: {param.shape}")
                print(f"    Mean: {param.data.mean().item():.6f}")
                print(f"    Std: {param.data.std().item():.6f}")
                print(f"    Min: {param.data.min().item():.6f}")
                print(f"    Max: {param.data.max().item():.6f}")
                
                if 'weight' in name:
                    classification_weights = param.data
                elif 'bias' in name:
                    classification_bias = param.data
                
                # Check if weights are biased toward class 0
                if param.data.dim() == 2 and param.data.shape[0] == 3:  # 3 classes
                    print(f"    Class 0 weight: {param.data[0].mean().item():.6f}")
                    print(f"    Class 1 weight: {param.data[1].mean().item():.6f}")
                    print(f"    Class 2 weight: {param.data[2].mean().item():.6f}")
                    
                    if param.data[0].mean() > param.data[1].mean() and param.data[0].mean() > param.data[2].mean():
                        print(f"    WARNING: Class 0 has higher average weights!")
        
        # Test 2: Check model predictions on random inputs
        print("\n2. CHECKING MODEL PREDICTIONS ON RANDOM INPUTS")
        print("-" * 50)
        
        # Create random inputs
        random_inputs = torch.randn(10, 3, 416, 416)
        
        with torch.no_grad():
            model_output = model(random_inputs)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            
            print(f"Classification output shape: {classification_output.shape}")
            print(f"Classification output:")
            print(f"  Mean: {classification_output.mean().item():.6f}")
            print(f"  Std: {classification_output.std().item():.6f}")
            
            # Check predictions
            probs = torch.softmax(classification_output, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            
            print(f"\nPredictions on random inputs:")
            class_names = ['A4C', 'PSAX', 'PLAX']
            for i, (pred_class, prob) in enumerate(zip(pred_classes, probs)):
                pred_name = class_names[pred_class.item()]
                confidence = prob[pred_class].item()
                print(f"  Sample {i}: {pred_name} (confidence: {confidence:.3f})")
            
            # Check if all predictions are class 0
            unique_preds = torch.unique(pred_classes)
            print(f"\nUnique predictions: {unique_preds.tolist()}")
            
            if len(unique_preds) == 1 and unique_preds[0] == 0:
                print(f"  ERROR: Model always predicts class 0!")
                return False
            else:
                print(f"  OK: Model predicts multiple classes")
        
        # Test 3: Check model predictions on uniform inputs
        print("\n3. CHECKING MODEL PREDICTIONS ON UNIFORM INPUTS")
        print("-" * 50)
        
        # Create uniform inputs (all zeros)
        uniform_inputs = torch.zeros(5, 3, 416, 416)
        
        with torch.no_grad():
            model_output = model(uniform_inputs)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            
            probs = torch.softmax(classification_output, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            
            print(f"Predictions on uniform (zero) inputs:")
            for i, (pred_class, prob) in enumerate(zip(pred_classes, probs)):
                pred_name = class_names[pred_class.item()]
                confidence = prob[pred_class].item()
                print(f"  Sample {i}: {pred_name} (confidence: {confidence:.3f})")
            
            # Check if all predictions are class 0
            unique_preds = torch.unique(pred_classes)
            print(f"\nUnique predictions: {unique_preds.tolist()}")
            
            if len(unique_preds) == 1 and unique_preds[0] == 0:
                print(f"  ERROR: Model always predicts class 0 even on uniform inputs!")
                return False
        
        # Test 4: Check if model was trained properly
        print("\n4. CHECKING MODEL TRAINING STATUS")
        print("-" * 40)
        
        # Check if model has been trained (non-zero gradients or updated weights)
        total_params = 0
        non_zero_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            non_zero_params += (param.data != 0).sum().item()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Non-zero parameters: {non_zero_params:,}")
        print(f"Percentage non-zero: {(non_zero_params/total_params)*100:.2f}%")
        
        if non_zero_params / total_params < 0.1:
            print(f"  WARNING: Very few non-zero parameters - model may not be trained!")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_initialization():
    """Check if model needs to be reinitialized"""
    print("\nCHECKING MODEL INITIALIZATION")
    print("=" * 40)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load a fresh model (not trained)
        print("Loading fresh model for comparison...")
        
        # Try to load the original model file
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        
        # Check if there's a backup or original model
        backup_paths = [
            "yolov5c/runs/classifybackbone13/weights/best.pt",
            "yolov5c/runs/classifybackbone13/weights/epoch_0.pt",
            "yolov5c/runs/classifybackbone13/weights/epoch_1.pt"
        ]
        
        for backup_path in backup_paths:
            if Path(backup_path).exists():
                print(f"Found backup model: {backup_path}")
                
                # Load backup model
                backup_model = attempt_load(backup_path, device='cpu', inplace=True, fuse=True)
                backup_model.eval()
                
                # Test predictions
                test_input = torch.randn(2, 3, 416, 416)
                
                with torch.no_grad():
                    backup_output = backup_model(test_input)
                
                if isinstance(backup_output, tuple) and len(backup_output) == 2:
                    detection_output, classification_output = backup_output
                    
                    probs = torch.softmax(classification_output, dim=-1)
                    pred_classes = torch.argmax(probs, dim=-1)
                    
                    print(f"Backup model predictions: {pred_classes.tolist()}")
                    
                    if len(torch.unique(pred_classes)) > 1:
                        print(f"  OK: Backup model predicts multiple classes")
                        return True
                    else:
                        print(f"  WARNING: Backup model also predicts only one class")
        
        print("No suitable backup models found")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main investigation function"""
    print("MODEL BIAS INVESTIGATION")
    print("=" * 60)
    
    # Test 1: Investigate model bias
    success1 = investigate_model_bias()
    
    # Test 2: Check model initialization
    success2 = check_model_initialization()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if not success1:
        print("ERROR: Model has prediction bias - always predicts class 0")
        print("   This explains the 40% accuracy and lack of learning")
        print("   Solution: Model needs retraining or weight reinitialization")
    else:
        print("SUCCESS: Model appears to work correctly")
        print("   The issue might be in training configuration or data")

if __name__ == "__main__":
    main()
