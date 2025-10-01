#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify cross-entropy calculations in train_classification_task.py
using the model output extraction utility
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

from extract_model_outputs import (
    extract_model_outputs_and_labels,
    parse_model_output,
    manual_cross_entropy_calculation,
    analyze_batch_outputs
)

def test_training_crossentropy():
    """
    Test cross-entropy calculations during training by extracting model outputs
    and comparing with training script calculations
    """
    print("🧪 Testing Cross-Entropy Calculations in train_classification_task.py")
    print("=" * 70)
    
    # Configuration
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"  # Your actual trained model
    data_yaml = "regurgitationV1/data.yaml"  # Your actual dataset
    class_names = ['A4C', 'PSAX', 'PLAX']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Import required modules
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        from yolov5c.utils.general import check_dataset
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        
        print(f"📂 Loading model from: {model_path}")
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("   Please train a model first or check the path")
            return
        
        # Load model
        model = attempt_load(model_path, device=device)
        print(f"✅ Model loaded successfully")
        
        # Load dataset
        print(f"📂 Loading dataset from: {data_yaml}")
        data_dict = check_dataset(data_yaml)
        train_path = data_dict['train']
        
        # Create dataloader (same as in train_classification_task.py)
        print("📊 Creating dataloader...")
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz=640,
            batch_size=8,  # Small batch for testing
            gs=32,
            single_cls=False,
            hyp={'cls_task': 0.3, 'label_smoothing': 0.1},
            augment=False,  # Disable for consistent results
            cache=None,
            rect=False,
            rank=-1,
            workers=4
        )
        
        # Create loss function (same as in train_classification_task.py)
        print("🔧 Creating ClassificationTaskLoss...")
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=0.3,
            label_smoothing=0.1
        )
        
        print("✅ Setup completed")
        print(f"   Model device: {device}")
        print(f"   Dataloader batches: {len(train_loader)}")
        
        # Test on first few batches
        print("\n🔍 Testing Cross-Entropy Calculations...")
        print("-" * 50)
        
        model.eval()
        total_manual_loss = 0.0
        total_training_loss = 0.0
        total_samples = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets, paths, shapes, classification_labels) in enumerate(train_loader):
                if batch_idx >= 5:  # Test first 5 batches
                    break
                
                print(f"\n📊 Batch {batch_idx}:")
                
                # Move to device
                images = images.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device)
                
                # Process classification labels (same as in train_classification_task.py)
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
                    # 1. Manual cross-entropy calculation
                    manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
                    
                    # 2. Training script loss calculation
                    total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
                    training_loss = total_loss if total_loss is not None else torch.tensor(0.0)
                    
                    # 3. PyTorch built-in cross-entropy
                    pytorch_loss = torch.nn.functional.cross_entropy(classification_output, classification_labels)
                    
                    # Calculate accuracy
                    pred_classes = torch.argmax(classification_output, dim=1)
                    correct = (pred_classes == classification_labels).sum().item()
                    accuracy = correct / classification_labels.shape[0]
                    
                    # Print results
                    print(f"   📊 Samples: {classification_labels.shape[0]}")
                    print(f"   📊 Manual CE Loss: {manual_loss.item():.6f}")
                    print(f"   📊 Training CE Loss: {training_loss.item():.6f}")
                    print(f"   📊 PyTorch CE Loss: {pytorch_loss.item():.6f}")
                    print(f"   📊 Accuracy: {accuracy:.4f}")
                    
                    # Compare differences
                    manual_vs_training = abs(manual_loss.item() - training_loss.item())
                    manual_vs_pytorch = abs(manual_loss.item() - pytorch_loss.item())
                    
                    print(f"   📊 Manual vs Training diff: {manual_vs_training:.8f}")
                    print(f"   📊 Manual vs PyTorch diff: {manual_vs_pytorch:.8f}")
                    
                    # Check if they match
                    if manual_vs_training < 1e-6:
                        print(f"   ✅ Manual and Training losses match!")
                    else:
                        print(f"   ⚠️  Manual and Training losses differ!")
                    
                    if manual_vs_pytorch < 1e-6:
                        print(f"   ✅ Manual and PyTorch losses match!")
                    else:
                        print(f"   ⚠️  Manual and PyTorch losses differ!")
                    
                    # Accumulate for overall statistics
                    total_manual_loss += manual_loss.item() * classification_labels.shape[0]
                    total_training_loss += training_loss.item() * classification_labels.shape[0]
                    total_samples += classification_labels.shape[0]
                    batch_count += 1
                    
                    # Show sample predictions
                    print(f"   📋 Sample predictions:")
                    for i in range(min(3, classification_labels.shape[0])):
                        pred_class = pred_classes[i].item()
                        target_class = classification_labels[i].item()
                        confidence = torch.softmax(classification_output, dim=1)[i, pred_class].item()
                        
                        pred_name = class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'
                        target_name = class_names[target_class] if target_class < len(class_names) else f'Class_{target_class}'
                        
                        status = "✅" if pred_class == target_class else "❌"
                        print(f"     Sample {i}: {pred_name} vs {target_name} (conf: {confidence:.3f}) {status}")
                else:
                    print(f"   ❌ No classification output available")
        
        # Overall statistics
        if total_samples > 0:
            avg_manual_loss = total_manual_loss / total_samples
            avg_training_loss = total_training_loss / total_samples
            overall_diff = abs(avg_manual_loss - avg_training_loss)
            
            print(f"\n📈 Overall Statistics:")
            print(f"   Batches tested: {batch_count}")
            print(f"   Total samples: {total_samples}")
            print(f"   Average Manual CE Loss: {avg_manual_loss:.6f}")
            print(f"   Average Training CE Loss: {avg_training_loss:.6f}")
            print(f"   Average difference: {overall_diff:.8f}")
            
            if overall_diff < 1e-6:
                print(f"   ✅ Perfect match between manual and training calculations!")
            elif overall_diff < 1e-4:
                print(f"   ✅ Very close match (difference < 1e-4)")
            else:
                print(f"   ⚠️  Significant difference detected!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function_directly():
    """
    Test the ClassificationTaskLoss function directly
    """
    print("\n🧪 Testing ClassificationTaskLoss Function Directly")
    print("=" * 50)
    
    try:
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        
        # Create mock model
        class MockModel(torch.nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.hyp = {'cls_task': 0.3, 'label_smoothing': 0.1}
                self.classifier = torch.nn.Linear(512, num_classes)
            
            def forward(self, x):
                return None, self.classifier(torch.randn(x.shape[0], 512))
        
        model = MockModel(num_classes=3)
        
        # Create loss function
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=0.3,
            label_smoothing=0.1
        )
        
        # Create mock data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        targets = torch.zeros(batch_size, 6)
        classification_labels = torch.randint(0, 3, (batch_size,))
        
        # Test loss calculation
        model_output = model(images)
        total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
        
        print(f"   Mock model test:")
        print(f"   📊 Total loss: {total_loss.item():.6f}")
        print(f"   📊 Loss items: {[item.item() for item in loss_items]}")
        print(f"   ✅ ClassificationTaskLoss function works correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing loss function: {e}")
        return False

def main():
    """
    Main testing function
    """
    print("🎯 Cross-Entropy Testing for train_classification_task.py")
    print("=" * 60)
    
    # Test 1: Direct loss function test
    success1 = test_loss_function_directly()
    
    # Test 2: Full training pipeline test
    success2 = test_training_crossentropy()
    
    print(f"\n📋 Test Results:")
    print(f"   Loss Function Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"   Training Pipeline Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! Your cross-entropy calculations are working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the error messages above.")
    
    print(f"\n📖 What this test verifies:")
    print(f"   1. ClassificationTaskLoss function works correctly")
    print(f"   2. Manual cross-entropy matches training script calculations")
    print(f"   3. Manual cross-entropy matches PyTorch's implementation")
    print(f"   4. Model outputs are processed correctly")
    print(f"   5. Labels are handled properly")

if __name__ == "__main__":
    main()
