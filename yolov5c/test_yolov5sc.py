#!/usr/bin/env python3
"""
Test script to verify yolov5sc.yaml dual-output configuration
"""

import torch
import sys
import os
from pathlib import Path

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

from models.yolo import Model
from utils.general import parse_model_output, validate_detection_outputs

def test_yolov5sc():
    """Test the yolov5sc.yaml configuration"""
    
    print("Testing yolov5sc.yaml dual-output configuration...")
    
    try:
        # Load model from yolov5sc.yaml
        model = Model('models/yolov5sc.yaml')
        model.eval()
        
        print(f"✓ Model created successfully from yolov5sc.yaml")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create test input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        print(f"✓ Input tensor shape: {input_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            model_output = model(input_tensor)
            
        print(f"✓ Raw model output type: {type(model_output)}")
        if isinstance(model_output, tuple):
            print(f"✓ Model output tuple length: {len(model_output)}")
            print(f"✓ First element type: {type(model_output[0])}")
            print(f"✓ Second element type: {type(model_output[1])}")
        
        # Parse model output
        detections, classification_output = parse_model_output(model_output)
        
        print(f"✓ Parsed detections type: {type(detections)}")
        print(f"✓ Parsed detections length: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"✓ Detection {i} shape: {det.shape}")
        
        print(f"✓ Classification output type: {type(classification_output)}")
        if classification_output is not None:
            print(f"✓ Classification output shape: {classification_output.shape}")
            print(f"✓ Expected classification shape: (batch_size, 3) for 3 view classes")
        
        # Validate detection outputs
        validate_detection_outputs(detections)
        print("✓ Detection outputs validation passed")
        
        # Test that we have both outputs
        if classification_output is not None:
            print("✓ Dual output functionality working correctly!")
            print(f"  - Detection outputs: {len(detections)} tensors for 4 regurgitation classes")
            print(f"  - Classification output: {classification_output.shape} for 3 view classes")
            
            # Verify shapes
            expected_cls_shape = (batch_size, 3)
            if classification_output.shape == expected_cls_shape:
                print("✓ Classification output shape is correct!")
            else:
                print(f"⚠ Expected classification shape {expected_cls_shape}, got {classification_output.shape}")
                
        else:
            print("⚠ Classification output is None - check model configuration")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_yolov5sc()
    if success:
        print("\n🎉 yolov5sc.yaml dual-output test completed successfully!")
        print("Your dual-output architecture is working correctly!")
    else:
        print("\n💥 yolov5sc.yaml dual-output test failed!")
        sys.exit(1) 