#!/usr/bin/env python3
"""
Quick test to verify yolov5sc.yaml model loads and runs forward pass
"""

import torch
import sys
import os
from pathlib import Path

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

def quick_test():
    """Quick test of model loading and forward pass"""
    
    print("Quick test of yolov5sc.yaml model...")
    
    try:
        # Import after adding to path
        from models.yolo import Model
        from utils.general import parse_model_output
        
        print("✓ Imports successful")
        
        # Load model
        model = Model('models/yolov5sc.yaml')
        print("✓ Model loaded successfully")
        
        # Create dummy input
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        print(f"✓ Input tensor created: {input_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            model_output = model(input_tensor)
        
        print("✓ Forward pass successful")
        
        # Parse output
        detections, classification_output = parse_model_output(model_output)
        
        print(f"✓ Output parsing successful")
        print(f"  - Detections: {len(detections)} tensors")
        print(f"  - Classification: {classification_output.shape if classification_output is not None else 'None'}")
        
        print("\n🎉 Quick test passed! Model is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        sys.exit(1) 