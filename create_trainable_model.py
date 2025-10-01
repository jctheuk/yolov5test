#!/usr/bin/env python3
"""
Create a trainable model from the frozen model
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def create_trainable_model():
    """Create a trainable model from the frozen model"""
    print("CREATING TRAINABLE MODEL")
    print("=" * 40)
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        # Load original model
        model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        
        print("OK: Original model loaded")
        
        # Unfreeze all parameters
        print("Unfreezing all parameters...")
        
        for name, param in model.named_parameters():
            param.requires_grad = True
        
        # Verify unfreezing
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        
        print(f"Parameters after unfreezing:")
        print(f"  Trainable: {trainable_params}")
        print(f"  Total: {total_params}")
        
        if trainable_params != total_params:
            print("ERROR: Not all parameters are trainable")
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
                print("SUCCESS: Gradients are flowing!")
                
                # Save the trainable model
                trainable_model_path = "yolov5c/runs/classifybackbone13/weights/last_trainable.pt"
                
                # Create proper checkpoint format
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
                
                torch.save(checkpoint, trainable_model_path)
                print(f"Trainable model saved to: {trainable_model_path}")
                
                return True
            else:
                print("ERROR: Still no gradients")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainable_model():
    """Test the trainable model on real data"""
    print("\nTESTING TRAINABLE MODEL ON REAL DATA")
    print("=" * 50)
    
    try:
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        
        # Load trainable model
        trainable_model_path = "yolov5c/runs/classifybackbone13/weights/last_trainable.pt"
        
        if not Path(trainable_model_path).exists():
            print(f"ERROR: Trainable model not found: {trainable_model_path}")
            return False
        
        # Load model manually to avoid the loading issue
        checkpoint = torch.load(trainable_model_path, map_location='cpu')
        model_state = checkpoint['model']
        
        # Load original model and replace state
        original_model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
        model = attempt_load(original_model_path, device='cpu', inplace=True, fuse=True)
        
        # Load the trainable state
        model.load_state_dict(model_state)
        
        # Ensure parameters are trainable
        for name, param in model.named_parameters():
            param.requires_grad = True
        
        print("OK: Trainable model loaded")
        
        # Test on real data
        print("Testing on real data...")
        
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
                print("SUCCESS: Trainable model works on real data!")
                
                # Test predictions
                with torch.no_grad():
                    probs = torch.softmax(classification_output, dim=-1)
                    pred_classes = torch.argmax(probs, dim=-1)
                    
                    class_names = ['A4C', 'PSAX', 'PLAX']
                    print(f"\nPredictions:")
                    for i, (pred_class, prob, true_label) in enumerate(zip(pred_classes, probs, classification_labels)):
                        pred_name = class_names[pred_class.item()]
                        confidence = prob[pred_class].item()
                        true_class = true_label.argmax().item()
                        true_name = class_names[true_class]
                        status = "✅" if pred_class.item() == true_class else "❌"
                        print(f"  Sample {i}: {pred_name} vs {true_name} (conf: {confidence:.3f}) {status}")
                
                return True
            else:
                print("ERROR: Trainable model still cannot learn")
                return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_command():
    """Create a training command for the trainable model"""
    print("\nCREATING TRAINING COMMAND")
    print("=" * 40)
    
    trainable_model_path = "yolov5c/runs/classifybackbone13/weights/last_trainable.pt"
    
    if Path(trainable_model_path).exists():
        training_command = f"""
# Training command for the trainable model
python train_classification_task.py \\
    --data regurgitationV1/data.yaml \\
    --epochs 50 \\
    --batch-size 16 \\
    --device auto \\
    --weights {trainable_model_path} \\
    --hyp yolov5c/runs/classifybackbone13/hyp.yaml
"""
        
        print("Training command:")
        print(training_command)
        
        # Save command to file
        with open("train_trainable_model.txt", "w") as f:
            f.write(training_command.strip())
        
        print("Training command saved to: train_trainable_model.txt")
        
        return True
    else:
        print("ERROR: Trainable model not found")
        return False

def main():
    """Main function"""
    print("CREATING TRAINABLE MODEL")
    print("=" * 60)
    
    # Step 1: Create trainable model
    success1 = create_trainable_model()
    
    if success1:
        # Step 2: Test trainable model
        success2 = test_trainable_model()
        
        if success2:
            # Step 3: Create training command
            success3 = create_training_command()
            
            if success3:
                print("\nFINAL SUMMARY:")
                print("=" * 40)
                print("SUCCESS: Trainable model created!")
                print("   - All parameters are trainable")
                print("   - Gradients are flowing")
                print("   - Model works on real data")
                print("   - Training command created")
                print("\nNext steps:")
                print("   1. Use 'last_trainable.pt' for training")
                print("   2. Run the training command")
                print("   3. Model should now achieve >40% accuracy")
                print("   4. Training should show learning progress")
            else:
                print("ERROR: Failed to create training command")
        else:
            print("ERROR: Trainable model testing failed")
    else:
        print("ERROR: Failed to create trainable model")

if __name__ == "__main__":
    main()
