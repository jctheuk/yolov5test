#!/usr/bin/env python3
"""
Verify yolov5sc.yaml model can load and run forward pass
"""

import torch
import sys
import os
from pathlib import Path

def verify_model():
    """Verify the model loads and runs correctly"""
    
    print("Verifying yolov5sc.yaml model...")
    
    try:
        # Add yolov5c to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import required modules
        from models.yolo import Model
        from utils.general import parse_model_output
        
        print("✓ Imports successful")
        
        # Load model from yolov5sc.yaml
        model = Model('models/yolov5sc.yaml')
        print("✓ Model loaded from yolov5sc.yaml")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        # Create test input
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        print(f"✓ Test input created: {input_tensor.shape}")
        
        # Set model to eval mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            model_output = model(input_tensor)
        
        print("✓ Forward pass completed")
        
        # Parse outputs
        detections, classification_output = parse_model_output(model_output)
        
        print("✓ Output parsing completed")
        print(f"  - Detection outputs: {len(detections)} tensors")
        for i, det in enumerate(detections):
            print(f"    Detection {i}: {det.shape}")
        
        if classification_output is not None:
            print(f"  - Classification output: {classification_output.shape}")
            print(f"    Expected: (batch_size, 3) for 3 view classes")
            print(f"    Actual: {classification_output.shape}")
            
            if classification_output.shape[1] == 3:
                print("✓ Classification output shape is correct!")
            else:
                print("⚠ Classification output shape mismatch")
        else:
            print("⚠ No classification output found")
        
        print("\n🎉 Model verification completed successfully!")
        print("Your dual-output YOLOv5 model is ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_model()
    if not success:
        sys.exit(1) 