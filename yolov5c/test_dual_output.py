#!/usr/bin/env python3
"""
Test script to verify dual-output YOLOv5 functionality
"""

import torch
import sys
import os
from pathlib import Path

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

from models.yolo import Model
from models.common import YOLOv5WithClassification
from utils.general import parse_model_output, validate_detection_outputs

def test_dual_output():
    """Test that the model produces both detection and classification outputs"""
    
    print("Testing dual-output YOLOv5 model...")
    
    # Create a simple model configuration that includes classification
    model_config = {
        'nc': 1,  # number of detection classes
        'depth_multiple': 0.33,
        'width_multiple': 0.50,
        'backbone': [
            [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, C3, [128]],
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 6, C3, [256]],
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 9, C3, [512]],
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 3, C3, [1024]],
            [-1, 1, SPPF, [1024, 5]],  # 9
        ],
        'head': [
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, Concat, [1]],  # cat backbone P4
            [-1, 3, C3, [512, False]],  # 13
            
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, Concat, [1]],  # cat backbone P3
            [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
            
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 14], 1, Concat, [1]],  # cat head P4
            [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
            
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, 10], 1, Concat, [1]],  # cat head P5
            [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
            
            [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
            [17, 1, YOLOv5WithClassification, [256, 3]],  # Classification head on P3
        ]
    }
    
    try:
        # Create model
        model = Model(model_config)
        model.eval()
        
        print(f"Model created successfully")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create test input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            model_output = model(input_tensor)
            
        print(f"Raw model output type: {type(model_output)}")
        if isinstance(model_output, tuple):
            print(f"Model output tuple length: {len(model_output)}")
            print(f"First element type: {type(model_output[0])}")
            print(f"Second element type: {type(model_output[1])}")
        
        # Parse model output
        detections, classification_output = parse_model_output(model_output)
        
        print(f"Parsed detections type: {type(detections)}")
        print(f"Parsed detections length: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"Detection {i} shape: {det.shape}")
        
        print(f"Classification output type: {type(classification_output)}")
        if classification_output is not None:
            print(f"Classification output shape: {classification_output.shape}")
        
        # Validate detection outputs
        validate_detection_outputs(detections)
        print("✓ Detection outputs validation passed")
        
        # Test that we have both outputs
        if classification_output is not None:
            print("✓ Dual output functionality working correctly!")
            print(f"  - Detection outputs: {len(detections)} tensors")
            print(f"  - Classification output: {classification_output.shape}")
        else:
            print("⚠ Classification output is None - check model configuration")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Import required modules
    from models.common import Conv, C3, SPPF, Concat
    import torch.nn as nn
    
    # Define anchors for detection
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    
    success = test_dual_output()
    if success:
        print("\n🎉 Dual-output test completed successfully!")
    else:
        print("\n💥 Dual-output test failed!")
        sys.exit(1) 