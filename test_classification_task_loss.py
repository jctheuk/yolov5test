#!/usr/bin/env python3
"""
Test script for ClassificationTaskLoss
Tests the classification task loss implementation for bugs
"""

import torch
import torch.nn as nn
import sys
import os

# Add yolov5c to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5c'))

try:
    from yolov5c.utils.classification_task_loss import ClassificationTaskLoss, SmartCrossEntropyLoss
    print("✅ Successfully imported ClassificationTaskLoss")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class MockModel(nn.Module):
    """Mock YOLOv5 model for testing"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.hyp = {'cls_task': 0.3, 'label_smoothing': 0.1}
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


def test_classification_task_loss():
    """Test the ClassificationTaskLoss functionality"""
    print("\n🧪 Testing ClassificationTaskLoss...")
    
    # Test parameters
    batch_size = 8
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create model and loss
    model = MockModel(num_classes).to(device)
    loss_fn = ClassificationTaskLoss(
        model, 
        enable_classification=True,
        cls_task_weight=0.3,
        label_smoothing=0.1
    )
    print("   ✅ Model and loss function created")
    
    # Create mock data
    pred_classification = torch.randn(batch_size, num_classes, device=device)
    targets_classification = torch.randint(0, num_classes, (batch_size,), device=device)
    print("   ✅ Mock data created")
    
    # Test loss computation
    preds = (None, pred_classification)  # (detection, classification)
    targets = (None, targets_classification)  # (detection, classification)
    
    total_loss, loss_items = loss_fn(preds, targets)
    print("   ✅ Loss computation successful")
    
    # Display results
    print(f"\n📊 Test Results:")
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Classification Loss: {loss_items[0].item():.4f}")
    print(f"   Classification Weight: {loss_fn.get_classification_weight():.3f}")
    print(f"   Enable Classification: {loss_fn.enable_classification}")
    
    # Test different epochs
    print(f"\n📈 Testing different epochs:")
    for epoch in [0, 5, 10, 15, 20]:
        loss_fn.set_epoch(epoch)
        weight = loss_fn.get_classification_weight()
        print(f"   Epoch {epoch:2d}: Classification weight = {weight:.3f}")
    
    # Test loss info
    loss_info = loss_fn.get_loss_info()
    print(f"\n📋 Loss Info: {loss_info}")
    
    print("\n✅ All tests passed successfully!")


def test_smart_cross_entropy_loss():
    """Test the SmartCrossEntropyLoss functionality"""
    print("\n🧪 Testing SmartCrossEntropyLoss...")
    
    batch_size = 8
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    criterion = SmartCrossEntropyLoss(label_smoothing=0.1)
    print("   ✅ SmartCrossEntropyLoss created")
    
    # Create mock data
    pred = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Test loss computation
    loss = criterion(pred, targets)
    print("   ✅ Loss computation successful")
    
    print(f"   Loss with label smoothing: {loss.item():.4f}")
    
    # Test without label smoothing
    criterion_no_smooth = SmartCrossEntropyLoss(label_smoothing=0.0)
    loss_no_smooth = criterion_no_smooth(pred, targets)
    print(f"   Loss without label smoothing: {loss_no_smooth.item():.4f}")
    
    print("   ✅ SmartCrossEntropyLoss tests passed!")


if __name__ == "__main__":
    print("🚀 Starting Classification Task Loss Tests...")
    
    try:
        test_classification_task_loss()
        test_smart_cross_entropy_loss()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
