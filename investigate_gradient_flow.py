#!/usr/bin/env python3
"""
Investigate why gradients are not flowing in the model
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def investigate_gradient_flow():
    """Investigate gradient flow issue"""
    print("INVESTIGATING GRADIENT FLOW ISSUE")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        print("OK: Model loaded successfully")
        
        # Test 1: Check if model parameters require gradients
        print("\n1. CHECKING PARAMETER GRADIENT REQUIREMENTS")
        print("-" * 50)
        
        total_params = 0
        requires_grad_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad:
                requires_grad_params += 1
            else:
                print(f"  Parameter {name} does NOT require gradients")
        
        print(f"Total parameters: {total_params}")
        print(f"Parameters requiring gradients: {requires_grad_params}")
        
        if requires_grad_params == 0:
            print("ERROR: No parameters require gradients!")
            return False
        elif requires_grad_params < total_params:
            print(f"WARNING: Only {requires_grad_params}/{total_params} parameters require gradients")
        else:
            print("OK: All parameters require gradients")
        
        # Test 2: Check if model is in training mode
        print("\n2. CHECKING MODEL TRAINING MODE")
        print("-" * 40)
        
        print(f"Model training mode: {model.training}")
        
        if not model.training:
            print("Setting model to training mode...")
            model.train()
            print(f"Model training mode after change: {model.training}")
        
        # Test 3: Test gradient flow with simple loss
        print("\n3. TESTING GRADIENT FLOW")
        print("-" * 30)
        
        # Create simple test data
        test_input = torch.randn(2, 3, 416, 416, requires_grad=True)
        test_targets = torch.tensor([0, 1], dtype=torch.long)
        
        print(f"Test input requires_grad: {test_input.requires_grad}")
        
        # Forward pass
        model_output = model(test_input)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            
            print(f"Classification output requires_grad: {classification_output.requires_grad}")
            print(f"Classification output grad_fn: {classification_output.grad_fn}")
            
            # Simple loss
            loss = torch.nn.CrossEntropyLoss()(classification_output, test_targets)
            print(f"Loss: {loss.item():.6f}")
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Loss grad_fn: {loss.grad_fn}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_count = 0
            total_grad_norm = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    
                    if grad_norm > 0:
                        print(f"  {name}: gradient norm = {grad_norm:.6f}")
            
            total_grad_norm = total_grad_norm ** 0.5
            
            print(f"\nGradient summary:")
            print(f"  Parameters with gradients: {grad_count}")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            
            if total_grad_norm > 0:
                print("OK: Gradients are flowing")
                return True
            else:
                print("ERROR: No gradients computed")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_parameter_freezing():
    """Check if model parameters are frozen"""
    print("\nCHECKING MODEL PARAMETER FREEZING")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        print("Checking parameter freezing...")
        
        frozen_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if not param.requires_grad:
                frozen_params += 1
                print(f"  FROZEN: {name}")
        
        print(f"\nFreezing summary:")
        print(f"  Total parameters: {total_params}")
        print(f"  Frozen parameters: {frozen_params}")
        print(f"  Trainable parameters: {total_params - frozen_params}")
        
        if frozen_params == total_params:
            print("ERROR: All parameters are frozen!")
            return False
        elif frozen_params > 0:
            print(f"WARNING: {frozen_params} parameters are frozen")
            return False
        else:
            print("OK: No parameters are frozen")
            return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_fresh_weights():
    """Test if model works with fresh weights"""
    print("\nTESTING MODEL WITH FRESH WEIGHTS")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        # Reset classification head weights
        print("Resetting classification head weights...")
        
        classification_head_found = False
        
        for name, param in model.named_parameters():
            if 'classify' in name.lower() or 'cls' in name.lower():
                classification_head_found = True
                print(f"  Resetting {name}")
                
                # Reset weights to small random values
                if param.dim() == 2:  # Linear layer weights
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif param.dim() == 1:  # Bias
                    torch.nn.init.zeros_(param)
        
        if not classification_head_found:
            print("ERROR: No classification head found!")
            return False
        
        # Test gradient flow after reset
        print("\nTesting gradient flow after weight reset...")
        
        model.train()
        test_input = torch.randn(2, 3, 416, 416, requires_grad=True)
        test_targets = torch.tensor([0, 1], dtype=torch.long)
        
        # Forward pass
        model_output = model(test_input)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_output, classification_output = model_output
            
            # Compute loss
            loss = torch.nn.CrossEntropyLoss()(classification_output, test_targets)
            print(f"Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm: {total_grad_norm:.6f}")
            
            if total_grad_norm > 0:
                print("SUCCESS: Gradients are flowing after weight reset!")
                return True
            else:
                print("ERROR: Still no gradients after weight reset")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main investigation function"""
    print("GRADIENT FLOW INVESTIGATION")
    print("=" * 60)
    
    # Test 1: Investigate gradient flow
    success1 = investigate_gradient_flow()
    
    # Test 2: Check parameter freezing
    success2 = check_model_parameter_freezing()
    
    # Test 3: Test with fresh weights
    success3 = test_model_with_fresh_weights()
    
    # Summary
    print("\nFINAL SUMMARY:")
    print("=" * 40)
    
    if success1:
        print("SUCCESS: Gradient flow is working")
    elif success2:
        print("ISSUE: Some parameters are frozen")
    elif success3:
        print("SOLUTION: Weight reset fixed gradient flow")
    else:
        print("ERROR: Gradient flow issue persists")
        print("   This explains why the model cannot learn")
        print("   Solution: Model needs retraining or weight reinitialization")

if __name__ == "__main__":
    main()
