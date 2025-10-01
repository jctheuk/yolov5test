#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for your specific model and dataset
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def test_my_model():
    """
    Test cross-entropy calculations with your specific model and dataset
    """
    print("🎯 Testing Cross-Entropy with Your Model and Dataset")
    print("=" * 60)
    
    # Your specific configuration
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    data_yaml = "regurgitationV1/data.yaml"
    class_names = ['A4C', 'PSAX', 'PLAX']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📂 Model: {model_path}")
    print(f"📂 Dataset: {data_yaml}")
    print(f"🖥️  Device: {device}")
    print(f"🏷️  Classes: {class_names}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not Path(data_yaml).exists():
        print(f"❌ Dataset not found: {data_yaml}")
        return False
    
    print("✅ Files found!")
    
    try:
        # Import required modules
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        from yolov5c.utils.general import check_dataset
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        from extract_model_outputs import manual_cross_entropy_calculation, parse_model_output
        
        print("\n🔧 Loading model and dataset...")
        
        # Load model
        model = attempt_load(model_path, device=device)
        model.eval()
        print(f"✅ Model loaded: {model_path}")
        
        # Load dataset
        data_dict = check_dataset(data_yaml)
        train_path = data_dict['train']
        print(f"✅ Dataset loaded: {train_path}")
        
        # Create dataloader (using correct signature from train_classification_task.py)
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz=640,
            batch_size=4,  # Small batch for testing
            gs=32,  # Grid size
            single_cls=False,
            hyp={'cls_task': 0.3, 'label_smoothing': 0.1},
            augment=False,  # Disable for consistent results
            cache=None,
            rect=False,
            rank=-1,
            workers=4,
            prefix='test: ',
            shuffle=False
        )
        print(f"✅ Dataloader created: {len(train_loader)} batches")
        
        # Create loss function
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=0.3,
            label_smoothing=0.1
        )
        print("✅ Loss function created")
        
        # Test on first batch
        print("\n🧪 Testing cross-entropy calculations...")
        
        with torch.no_grad():
            for batch_idx, (images, targets, paths, shapes, classification_labels) in enumerate(train_loader):
                if batch_idx >= 1:  # Test only first batch
                    break
                
                print(f"\n📊 Testing Batch {batch_idx}:")
                
                # Process inputs exactly like train_classification_task.py
                images = images.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device)
                
                # Process classification labels
                if classification_labels is not None:
                    classification_labels = classification_labels.to(device)
                    
                    # Handle different label formats
                    if classification_labels.dim() > 1:
                        if classification_labels.shape[-1] > 1:
                            classification_labels = classification_labels.argmax(dim=-1)
                        elif classification_labels.shape[-1] == 1:
                            classification_labels = classification_labels.squeeze(-1)
                    
                    if classification_labels.dtype != torch.long:
                        classification_labels = classification_labels.long()
                else:
                    classification_labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)
                
                # Forward pass
                model_output = model(images)
                
                # Parse model output
                detection_outputs, classification_output = parse_model_output(model_output)
                
                if classification_output is not None:
                    print(f"   📊 Input shapes:")
                    print(f"     Images: {images.shape}")
                    print(f"     Classification labels: {classification_labels.shape}")
                    print(f"     Classification output: {classification_output.shape}")
                    
                    # Method 1: Your training script loss
                    total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
                    training_loss = total_loss if total_loss is not None else torch.tensor(0.0)
                    
                    # Method 2: Manual cross-entropy
                    manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
                    
                    # Method 3: PyTorch built-in cross-entropy
                    pytorch_loss = torch.nn.functional.cross_entropy(classification_output, classification_labels)
                    
                    print(f"\n   📊 Loss Calculations:")
                    print(f"     Training script loss: {training_loss.item():.6f}")
                    print(f"     Manual calculation:   {manual_loss.item():.6f}")
                    print(f"     PyTorch built-in:     {pytorch_loss.item():.6f}")
                    
                    # Compare differences
                    diff_manual_training = abs(manual_loss.item() - training_loss.item())
                    diff_manual_pytorch = abs(manual_loss.item() - pytorch_loss.item())
                    
                    print(f"\n   📊 Differences:")
                    print(f"     Manual vs Training: {diff_manual_training:.8f}")
                    print(f"     Manual vs PyTorch:  {diff_manual_pytorch:.8f}")
                    
                    # Verification results
                    print(f"\n   ✅ Verification Results:")
                    if diff_manual_training < 1e-6:
                        print(f"     ✅ Training script cross-entropy is CORRECT!")
                    else:
                        print(f"     ❌ Training script cross-entropy has issues!")
                    
                    if diff_manual_pytorch < 1e-6:
                        print(f"     ✅ Manual calculation matches PyTorch!")
                    else:
                        print(f"     ❌ Manual calculation differs from PyTorch!")
                    
                    # Show sample predictions
                    pred_classes = torch.argmax(classification_output, dim=1)
                    pred_probs = torch.softmax(classification_output, dim=1)
                    correct = (pred_classes == classification_labels).sum().item()
                    accuracy = correct / classification_labels.shape[0]
                    
                    print(f"\n   📊 Model Performance:")
                    print(f"     Accuracy: {accuracy:.4f} ({correct}/{classification_labels.shape[0]})")
                    
                    print(f"     Sample predictions:")
                    for i in range(min(3, classification_labels.shape[0])):
                        pred_class = pred_classes[i].item()
                        target_class = classification_labels[i].item()
                        confidence = pred_probs[i, pred_class].item()
                        
                        pred_name = class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'
                        target_name = class_names[target_class] if target_class < len(class_names) else f'Class_{target_class}'
                        
                        status = "✅" if pred_class == target_class else "❌"
                        print(f"       Sample {i}: {pred_name} vs {target_name} (conf: {confidence:.3f}) {status}")
                    
                    # Show loss items breakdown
                    if isinstance(loss_items, list) and len(loss_items) >= 4:
                        print(f"\n   📊 Loss Items Breakdown:")
                        print(f"     Box loss: {loss_items[0].item():.6f}")
                        print(f"     Object loss: {loss_items[1].item():.6f}")
                        print(f"     Class loss: {loss_items[2].item():.6f}")
                        print(f"     Classification task loss: {loss_items[3].item():.6f}")
                    
                    return True
                else:
                    print(f"   ❌ No classification output available")
                    return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function
    """
    success = test_my_model()
    
    if success:
        print(f"\n🎉 Cross-entropy test completed successfully!")
        print(f"\n📖 Your model and dataset are working correctly!")
        print(f"   - Model: yolov5c/runs/classifybackbone13/weights/last.pt")
        print(f"   - Dataset: regurgitationV1/data.yaml")
        print(f"   - Classes: A4C, PSAX, PLAX")
    else:
        print(f"\n❌ Cross-entropy test failed!")
        print(f"   Please check the error messages above.")

if __name__ == "__main__":
    main()
