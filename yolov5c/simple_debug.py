#!/usr/bin/env python3
"""
Simple Training Diagnostic Script
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
import yaml

def main():
    print("🔍 Starting Simple Diagnostic...")
    
    try:
        # Test basic imports
        print("✅ Basic imports successful")
        
        # Test device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        
        # Test model loading
        from models.yolo import DetectionModel
        print("✅ Model import successful")
        
        # Load model
        model_cfg = 'models/yolov5sc.yaml'
        model = DetectionModel(model_cfg, ch=3, nc=4)
        model = model.to(device)
        print("✅ Model loaded successfully")
        
        # Test loss function
        from utils.loss import ComputeLoss
        loss_fn = ComputeLoss(model, autobalance=False)
        print("✅ Loss function loaded successfully")
        
        # Test hyperparameters
        hyp_yaml = 'data/hyps/hyp.fixed_classification_minimal.yaml'
        with open(hyp_yaml) as f:
            hyp = yaml.safe_load(f)
        print("✅ Hyperparameters loaded successfully")
        
        # Test optimizer
        from utils.torch_utils import smart_optimizer
        optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], momentum=hyp['momentum'], 
                                   decay=hyp['weight_decay'])
        print("✅ Optimizer created successfully")
        
        # Test data loading
        data_yaml = '../Regurgitation-YOLODataset-Detection/data.yaml'
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        print("✅ Data configuration loaded successfully")
        
        # Create diagnostic data
        diagnostic_data = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'device': str(device)
            },
            'hyperparameters': hyp,
            'data_config': data
        }
        
        # Collect bias data
        bias_data = {}
        for i, m in enumerate(model.model):
            if hasattr(m, 'm') and hasattr(m, 'stride'):  # Detect layer
                for j, mi in enumerate(m.m):
                    if hasattr(mi, 'bias') and mi.bias is not None:
                        b = mi.bias.view(m.na, -1)
                        bias_data[f'layer_{i}_conv_{j}'] = {
                            'objectness_bias': b.data[:, 4].cpu().numpy().tolist(),
                            'classification_bias': b.data[:, 5:5+m.nc].cpu().numpy().tolist(),
                            'stride': m.stride[j].item() if hasattr(m.stride, '__getitem__') else m.stride,
                            'bias_mean': mi.bias.mean().item(),
                            'bias_std': mi.bias.std().item()
                        }
        
        diagnostic_data['bias_data'] = bias_data
        
        # Save diagnostic data
        save_path = Path('diagnostic_results')
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / 'simple_diagnostic.json', 'w') as f:
            json.dump(diagnostic_data, f, indent=2)
        
        print("✅ Diagnostic data saved successfully")
        
        # Print key findings
        print("\n📊 Key Findings:")
        print(f"- Total parameters: {diagnostic_data['model_info']['total_parameters']:,}")
        print(f"- Trainable parameters: {diagnostic_data['model_info']['trainable_parameters']:,}")
        
        if bias_data:
            print("\n🔍 Bias Analysis:")
            for layer_name, bias_info in bias_data.items():
                print(f"  {layer_name}:")
                print(f"    Objectness bias mean: {np.mean(bias_info['objectness_bias']):.4f}")
                print(f"    Objectness bias std: {np.std(bias_info['objectness_bias']):.4f}")
                print(f"    Classification bias mean: {np.mean(bias_info['classification_bias']):.4f}")
                print(f"    Stride: {bias_info['stride']}")
        
        print("\n✅ Simple diagnostic complete!")
        
    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
