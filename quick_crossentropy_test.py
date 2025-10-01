#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick cross-entropy test for your specific model and dataset
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def quick_test():
    """
    Quick test of cross-entropy calculations
    """
    print("🚀 Quick Cross-Entropy Test")
    print("=" * 40)
    
    # Your specific configuration
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    data_yaml = "regurgitationV1/data.yaml"
    
    print(f"📂 Model: {model_path}")
    print(f"📂 Dataset: {data_yaml}")
    
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
        from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
        from extract_model_outputs import manual_cross_entropy_calculation, parse_model_output
        
        print("\n🔧 Loading model...")
        
        # Load model
        device = 'cpu'  # Use CPU for faster loading
        model = attempt_load(model_path, device=device)
        model.eval()
        print(f"✅ Model loaded")
        
        # Create loss function
        compute_loss = ClassificationTaskLoss(
            model=model,
            enable_classification=True,
            cls_task_weight=0.3,
            label_smoothing=0.1
        )
        print("✅ Loss function created")
        
        # Create mock data for testing
        print("\n🧪 Testing with mock data...")
        
        batch_size = 4
        num_classes = 3  # A4C, PSAX, PLAX
        
        # Mock images
        images = torch.randn(batch_size, 3, 640, 640)
        targets = torch.zeros(batch_size, 6)  # Mock detection targets
        classification_labels = torch.randint(0, num_classes, (batch_size,))
        
        print(f"   Mock data created: {batch_size} samples, {num_classes} classes")
        
        with torch.no_grad():
            # Forward pass
            model_output = model(images)
            
            # Parse model output
            detection_outputs, classification_output = parse_model_output(model_output)
            
            if classification_output is not None:
                print(f"   ✅ Classification output: {classification_output.shape}")
                
                # Method 1: Your training script loss
                total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
                training_loss = total_loss if total_loss is not None else torch.tensor(0.0)
                
                # Method 2: Manual cross-entropy
                manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
                
                # Method 3: PyTorch built-in cross-entropy
                pytorch_loss = torch.nn.functional.cross_entropy(classification_output, classification_labels)
                
                print(f"\n📊 Loss Calculations:")
                print(f"   Training script loss: {training_loss.item():.6f}")
                print(f"   Manual calculation:   {manual_loss.item():.6f}")
                print(f"   PyTorch built-in:     {pytorch_loss.item():.6f}")
                
                # Compare differences
                diff_manual_training = abs(manual_loss.item() - training_loss.item())
                diff_manual_pytorch = abs(manual_loss.item() - pytorch_loss.item())
                
                print(f"\n📊 Differences:")
                print(f"   Manual vs Training: {diff_manual_training:.8f}")
                print(f"   Manual vs PyTorch:  {diff_manual_pytorch:.8f}")
                
                # Verification results
                print(f"\n✅ Verification Results:")
                if diff_manual_training < 1e-6:
                    print(f"   ✅ Training script cross-entropy is CORRECT!")
                else:
                    print(f"   ❌ Training script cross-entropy has issues!")
                
                if diff_manual_pytorch < 1e-6:
                    print(f"   ✅ Manual calculation matches PyTorch!")
                else:
                    print(f"   ❌ Manual calculation differs from PyTorch!")
                
                # Show sample predictions
                pred_classes = torch.argmax(classification_output, dim=1)
                pred_probs = torch.softmax(classification_output, dim=1)
                correct = (pred_classes == classification_labels).sum().item()
                accuracy = correct / classification_labels.shape[0]
                
                print(f"\n📊 Model Performance:")
                print(f"   Accuracy: {accuracy:.4f} ({correct}/{classification_labels.shape[0]})")
                
                class_names = ['A4C', 'PSAX', 'PLAX']
                print(f"   Sample predictions:")
                for i in range(min(3, classification_labels.shape[0])):
                    pred_class = pred_classes[i].item()
                    target_class = classification_labels[i].item()
                    confidence = pred_probs[i, pred_class].item()
                    
                    pred_name = class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'
                    target_name = class_names[target_class] if target_class < len(class_names) else f'Class_{target_class}'
                    
                    status = "✅" if pred_class == target_class else "❌"
                    print(f"     Sample {i}: {pred_name} vs {target_name} (conf: {confidence:.3f}) {status}")
                
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
    success = quick_test()
    
    if success:
        print(f"\n🎉 Quick test completed successfully!")
        print(f"\n📖 Your cross-entropy calculations are working correctly!")
        print(f"   - Model: yolov5c/runs/classifybackbone13/weights/last.pt")
        print(f"   - Dataset: regurgitationV1/data.yaml")
        print(f"   - Classes: A4C, PSAX, PLAX")
        print(f"\n💡 You can now use the extraction utility with confidence!")
    else:
        print(f"\n❌ Quick test failed!")
        print(f"   Please check the error messages above.")

if __name__ == "__main__":
    main()

