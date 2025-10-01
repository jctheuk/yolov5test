#!/usr/bin/env python3
"""
Fix frozen parameters in the model
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def fix_frozen_parameters():
    """Fix frozen parameters in the model"""
    print("FIXING FROZEN PARAMETERS")
    print("=" * 40)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        print("OK: Model loaded successfully")
        
        # Unfreeze all parameters
        print("Unfreezing all parameters...")
        
        for name, param in model.named_parameters():
            param.requires_grad = True
        
        # Verify unfreezing
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += 1
            else:
                frozen_params += 1
        
        print(f"Parameters after unfreezing:")
        print(f"  Trainable: {trainable_params}")
        print(f"  Frozen: {frozen_params}")
        
        if frozen_params == 0:
            print("SUCCESS: All parameters are now trainable!")
        else:
            print(f"WARNING: {frozen_params} parameters are still frozen")
            return False
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        
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
            grad_count = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    grad_count += 1
            
            total_grad_norm = total_grad_norm ** 0.5
            
            print(f"Gradient summary:")
            print(f"  Parameters with gradients: {grad_count}")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            
            if total_grad_norm > 0:
                print("SUCCESS: Gradients are now flowing!")
                return True
            else:
                print("ERROR: Still no gradients")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_fixed_model():
    """Save the model with unfrozen parameters"""
    print("\nSAVING FIXED MODEL")
    print("=" * 30)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        # Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
        
        # Save fixed model
        fixed_model_path = "yolov5c/runs/classifybackbone13/weights/last_fixed.pt"
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': None,
            'epoch': -1,
            'best_fitness': None,
            'hyp': getattr(model, 'hyp', {}),
            'results': None,
            'date': None,
            'version': None
        }
        
        torch.save(checkpoint, fixed_model_path)
        print(f"Fixed model saved to: {fixed_model_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_model():
    """Test the fixed model"""
    print("\nTESTING FIXED MODEL")
    print("=" * 30)
    
    try:
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        
        # Load fixed model
        fixed_model_path = "yolov5c/runs/classifybackbone13/weights/last_fixed.pt"
        
        if not Path(fixed_model_path).exists():
            print(f"ERROR: Fixed model not found: {fixed_model_path}")
            return False
        
        model = attempt_load(fixed_model_path, device='cpu', inplace=True, fuse=True)
        
        # Check if parameters are unfrozen
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        
        print(f"Fixed model parameters:")
        print(f"  Trainable: {trainable_params}")
        print(f"  Total: {total_params}")
        
        if trainable_params == total_params:
            print("SUCCESS: Fixed model has all trainable parameters!")
            
            # Test on real data
            print("\nTesting on real data...")
            
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
                shuffle=True
            )
            
            batch = next(iter(train_loader))
            images, targets, paths, shapes, classification_labels = batch
            
            model.train()
            
            # Forward pass
            model_output = model(images)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                detection_output, classification_output = model_output
                
                # Compute loss
                target_indices = classification_labels.argmax(dim=-1).long()
                loss = torch.nn.CrossEntropyLoss()(classification_output, target_indices)
                
                print(f"Loss: {loss.item():.6f}")
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                total_grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                
                total_grad_norm = total_grad_norm ** 0.5
                print(f"Gradient norm: {total_grad_norm:.6f}")
                
                if total_grad_norm > 0:
                    print("SUCCESS: Fixed model can learn!")
                    return True
                else:
                    print("ERROR: Fixed model still cannot learn")
                    return False
            
        else:
            print("ERROR: Fixed model still has frozen parameters")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main fix function"""
    print("FIXING FROZEN PARAMETERS")
    print("=" * 60)
    
    # Step 1: Fix frozen parameters
    success1 = fix_frozen_parameters()
    
    if success1:
        # Step 2: Save fixed model
        success2 = save_fixed_model()
        
        if success2:
            # Step 3: Test fixed model
            success3 = test_fixed_model()
            
            if success3:
                print("\nFINAL SUMMARY:")
                print("=" * 40)
                print("SUCCESS: Model parameters fixed!")
                print("   - All parameters are now trainable")
                print("   - Gradients are flowing")
                print("   - Model can learn")
                print("   - Fixed model saved as 'last_fixed.pt'")
                print("\nNext steps:")
                print("   1. Use 'last_fixed.pt' for training")
                print("   2. Model should now achieve >40% accuracy")
                print("   3. Training should show learning progress")
            else:
                print("ERROR: Fixed model testing failed")
        else:
            print("ERROR: Failed to save fixed model")
    else:
        print("ERROR: Failed to fix frozen parameters")

if __name__ == "__main__":
    main()
