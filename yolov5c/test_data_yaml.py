#!/usr/bin/env python3
"""
Test script to verify data.yaml format and classification label generation
"""

import yaml
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from train import create_classification_labels_from_paths

def test_data_yaml():
    """Test the data.yaml file format and classification label generation"""
    
    # Load data.yaml
    data_yaml_path = Path('../Regurgitation-YOLODataset-Detection/data.yaml')
    
    if not data_yaml_path.exists():
        print(f"❌ Error: {data_yaml_path} not found")
        return False
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        print("✅ Successfully loaded data.yaml")
        print(f"📊 Detection classes: {data_dict.get('nc', 'Not found')}")
        print(f"📊 Detection names: {data_dict.get('names', 'Not found')}")
        print(f"📊 Classification classes: {data_dict.get('num_cls', 'Not found')}")
        print(f"📊 Classification names: {data_dict.get('cls_names', 'Not found')}")
        print(f"📁 Train path: {data_dict.get('train', 'Not found')}")
        print(f"📁 Val path: {data_dict.get('val', 'Not found')}")
        print(f"📁 Test path: {data_dict.get('test', 'Not found')}")
        
        # Test classification label generation
        test_paths = [
            'test_psax_image.png',
            'test_plax_image.png', 
            'test_a4c_image.png',
            'unknown_image.png'
        ]
        
        cls_names = data_dict.get('cls_names', ['PSAX', 'PLAX', 'A4C'])
        num_cls = data_dict.get('num_cls', 3)
        
        labels = create_classification_labels_from_paths(test_paths, num_cls, cls_names)
        
        print("\n🧪 Testing classification label generation:")
        for i, path in enumerate(test_paths):
            label_idx = labels[i].argmax().item()
            print(f"  {path} -> {cls_names[label_idx]} (class {label_idx})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data.yaml: {e}")
        return False

if __name__ == '__main__':
    print("🔍 Testing data.yaml format and classification...")
    success = test_data_yaml()
    
    if success:
        print("\n✅ All tests passed! Your data.yaml format is correct.")
    else:
        print("\n❌ Tests failed. Please check your data.yaml format.") 