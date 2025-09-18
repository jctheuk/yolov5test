#!/usr/bin/env python3
"""
Test script to verify YOLOv5WithClassification model configuration
"""

import torch
import sys
import os

# Add yolov5c to path
sys.path.append('yolov5c')

from models.yolo import DetectionModel
from models.common import YOLOv5WithClassification
from utils.general import check_yaml

def test_model_config():
    """Test the model configuration and classification layer"""
    
    print("🔍 Testing YOLOv5WithClassification Model Configuration")
    print("=" * 60)
    
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Current directory:", os.getcwd())
    
    # Load model configuration
    cfg_path = 'yolov5c/models/yolov5sc.yaml'
    print(f"📁 Loading config: {cfg_path}")
    
    try:
        # Create model
        model = DetectionModel(cfg_path, ch=3, nc=4)
        print("✅ Model created successfully")
        
        # Check if classification layer exists
        classification_layers = []
        for i, m in enumerate(model.model):
            if isinstance(m, YOLOv5WithClassification):
                classification_layers.append((i, m))
        
        print(f"🔍 Found {len(classification_layers)} classification layer(s)")
        
        for layer_idx, layer in classification_layers:
            print(f"  Layer {layer_idx}: {layer}")
            print(f"    Input channels: {layer.num_classes if hasattr(layer, 'num_classes') else 'Unknown'}")
            print(f"    Output classes: {layer.num_classes}")
        
        # Test forward pass with dummy input
        print("\n🧪 Testing forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            detection_outputs, classification_output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"  Detection outputs: {len(detection_outputs) if detection_outputs else 0} tensors")
        if detection_outputs:
            for i, det in enumerate(detection_outputs):
                print(f"    Detection {i}: {det.shape}")
        
        if classification_output is not None:
            print(f"  Classification output: {classification_output.shape}")
            print(f"    Expected: [batch_size, 3] (A4C, PSAX, PLAX)")
            print(f"    Actual: {classification_output.shape}")
            
            if classification_output.shape[1] == 3:
                print("✅ Classification output has correct number of classes (3)")
            else:
                print(f"❌ Classification output has wrong number of classes: {classification_output.shape[1]}")
        else:
            print("❌ No classification output found")
        
        # Print model summary
        print(f"\n📊 Model Summary:")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting test...")
    test_model_config()
    print("Test completed.")
