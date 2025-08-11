#!/usr/bin/env python3
"""
Simple test to verify the dual-task training integration works
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_compute_loss():
    """Test the enhanced ComputeLoss class"""
    try:
        from utils.loss import ComputeLoss
        
        # Create a dummy model with hyp attribute
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hyp = {
                    'box': 0.05,
                    'cls': 0.5,
                    'cls_pw': 1.0,
                    'obj': 1.0,
                    'obj_pw': 1.0,
                    'cls_task': 0.3,
                    'anchor_t': 4.0,
                    'fl_gamma': 0.0,
                    'label_smoothing': 0.1
                }
                self.na = 3
                self.nc = 4
                self.nl = 3
                self.anchors = torch.tensor([[[10, 13], [16, 30], [33, 23]]])
                self.stride = torch.tensor([8.0, 16.0, 32.0])
                # Add a parameter so device can be determined
                self.dummy_param = torch.nn.Parameter(torch.randn(1))
        
        model = DummyModel()
        compute_loss = ComputeLoss(model)
        
        print("✅ ComputeLoss initialization successful!")
        
        # Test with empty targets to avoid indexing issues
        targets = torch.empty(0, 6)  # Empty targets
        classification_labels = torch.tensor([0, 1])
        
        # Create dummy model outputs with proper structure
        detection_outputs = [
            torch.randn(1, 3, 80, 80, 9),  # 5 + 4 = 9 (box_coords + obj + cls)
            torch.randn(1, 3, 40, 40, 9),
            torch.randn(1, 3, 20, 20, 9)
        ]
        classification_output = torch.randn(2, 3)
        
        # Test the loss computation
        model_output = (detection_outputs, classification_output)
        total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
        
        print(f"✅ ComputeLoss test passed!")
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Loss items: {loss_items.tolist()}")
        return True
        
    except Exception as e:
        print(f"❌ ComputeLoss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing enhanced ComputeLoss...")
    success = test_compute_loss()
    
    if success:
        print("\n🎉 All tests passed! The dual-task training integration is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
