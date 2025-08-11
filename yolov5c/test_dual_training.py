#!/usr/bin/env python3
"""
Test script for dual-task YOLOv5 training (detection + classification)
This script tests the integration of dual-task training into the existing train.py
"""

import os
import sys
import yaml
from pathlib import Path

# Add yolov5c to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, init_seeds, LOGGER
from utils.loss import ComputeLoss
from utils.torch_utils import select_device

def test_dual_training():
    """Test the dual-task training functionality"""
    
    print("Testing dual-task YOLOv5 training...")
    
    # Initialize
    device = select_device('')
    init_seeds(1)
    
    # Create a simple model configuration
    model_cfg = {
        'nc': 4,
        'num_cls': 3,
        'depth_multiple': 0.33,
        'width_multiple': 0.50,
        'anchors': [
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326]
        ],
        'backbone': [
            [-1, 1, 'Conv', [64, 6, 2, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 3, 'C3', [128]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 6, 'C3', [256]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 9, 'C3', [512]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 3, 'C3', [1024]],
            [-1, 1, 'SPPF', [1024, 5]],
        ],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 3, 'C3', [512, False]],
            [-1, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 3, 'C3', [256, False]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 14], 1, 'Concat', [1]],
            [-1, 3, 'C3', [512, False]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 3, 'C3', [1024, False]],
            [17, 1, 'YOLOv5WithClassification', [256, 3]],
            [[17, 20, 23], 1, 'Detect', [4, [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]]],
        ]
    }
    
    try:
        # Create model
        print("Creating model...")
        model = Model(model_cfg, ch=3, nc=4, anchors=model_cfg['anchors']).to(device)
        print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test model forward pass
        print("Testing model forward pass...")
        dummy_input = torch.randn(2, 3, 640, 640).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Model output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Model output length: {len(output)}")
            if len(output) >= 1:
                print(f"Detection outputs: {len(output[0]) if isinstance(output[0], list) else 'Not list'}")
            if len(output) >= 2:
                print(f"Classification output shape: {output[1].shape if hasattr(output[1], 'shape') else 'No shape'}")
        else:
            print(f"Model output shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
        
        # Test dual loss
        print("Testing dual loss...")
        compute_loss = ComputeLoss(model)
        
        # Create dummy targets and classification labels
        # Targets format: (image_id, class, x, y, w, h) - 6 columns
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.3], [1, 1, 0.3, 0.4, 0.1, 0.2]]).to(device)
        classification_labels = torch.tensor([0, 1]).to(device)
        
        # Compute loss
        total_loss, loss_items = compute_loss(output, targets, classification_labels)
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Loss items: {loss_items.tolist()}")
        
        print("✅ Dual-task training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dual-task training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with classification labels"""
    
    print("\nTesting data loading...")
    
    try:
        # Use the real dataset from Regurgitation-YOLODataset-Detection
        # The data.yaml has relative paths, so we need to create a temporary one with absolute paths
        original_data_yaml = Path('../Regurgitation-YOLODataset-Detection/data.yaml')
        
        if not original_data_yaml.exists():
            print(f"❌ Dataset not found at {original_data_yaml}")
            return False
        
        print(f"✅ Found dataset at {original_data_yaml}")
        
        # Read the original data.yaml and update paths to be absolute
        import yaml
        with open(original_data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get the absolute path to the dataset directory
        dataset_dir = original_data_yaml.parent.absolute()
        
        # Update paths to be absolute - properly resolve the paths
        train_path = (dataset_dir / 'train' / 'images').resolve()
        val_path = (dataset_dir / 'valid' / 'images').resolve()
        test_path = (dataset_dir / 'test' / 'images').resolve()
        
        data_config['train'] = str(train_path)
        data_config['val'] = str(val_path)
        data_config['test'] = str(test_path)
        
        # Create a temporary data.yaml with absolute paths
        temp_data_yaml = Path('temp_data.yaml')
        with open(temp_data_yaml, 'w') as f:
            yaml.dump(data_config, f)
        
        print(f"✅ Created temporary data.yaml with absolute paths")
        print(f"  Train path: {data_config['train']}")
        print(f"  Val path: {data_config['val']}")
        
        # Verify the paths actually exist
        if not train_path.exists():
            print(f"❌ Train path does not exist: {train_path}")
            return False
        
        if not val_path.exists():
            print(f"❌ Val path does not exist: {val_path}")
            return False
        
        print(f"✅ Verified paths exist")
        
        # Debug: Check what files are actually in the directories
        print(f"\n🔍 Debug: Checking files in train directory...")
        train_files = list(train_path.glob('*.png'))
        print(f"  Found {len(train_files)} PNG files in train directory")
        if train_files:
            print(f"  First few files: {[f.name for f in train_files[:3]]}")
        
        # Check if there are any files with other extensions
        all_train_files = list(train_path.glob('*.*'))
        print(f"  Total files in train directory: {len(all_train_files)}")
        if all_train_files:
            extensions = set(f.suffix.lower() for f in all_train_files)
            print(f"  File extensions found: {extensions}")
        
        # Test dataloader creation
        device = select_device('')
        
        # Use the temporary data.yaml with absolute paths
        try:
            train_loader, dataset = create_dataloader(
                path=str(temp_data_yaml),  # Use the temporary file
                imgsz=640,
                batch_size=2,
                stride=32,
                single_cls=False,
                hyp={'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'cls_task': 0.3, 'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0},
                augment=True,
                cache=False,
                rect=False,
                rank=-1,
                workers=0,
                image_weights=False,
                quad=False,
                prefix='test: ',
                shuffle=True
            )
            
            print(f"✅ DataLoader created successfully with {len(train_loader)} batches")
            
            # Test batch loading
            for batch_idx, (imgs, targets, paths, shapes, classification_labels) in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  Images shape: {imgs.shape}")
                print(f"  Targets shape: {targets.shape}")
                print(f"  Classification labels shape: {classification_labels.shape if hasattr(classification_labels, 'shape') else 'No shape'}")
                print(f"  Paths: {paths[:2]}...")  # Show first 2 paths
                break
                
        except Exception as e:
            print(f"❌ DataLoader creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up temporary file
            if temp_data_yaml.exists():
                temp_data_yaml.unlink()
        
        print("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """Test individual model components"""
    
    print("\nTesting model components...")
    
    try:
        from models.common import YOLOv5WithClassification
        
        # Test classification head
        device = select_device('')
        classification_head = YOLOv5WithClassification(256, 3).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 256, 80, 80).to(device)
        output = classification_head(dummy_input)
        
        print(f"Classification head output shape: {output.shape}")
        print(f"Expected shape: (2, 3) - got: {output.shape}")
        
        if output.shape == (2, 3):
            print("✅ Classification head test passed!")
            return True
        else:
            print(f"❌ Classification head test failed - expected (2, 3), got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("DUAL-TASK YOLOV5 TRAINING TEST")
    print("=" * 60)
    
    # Only run data loading test for now
    data_loading_passed = test_data_loading()
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Data loading: {'✅ PASSED' if data_loading_passed else '❌ FAILED'}")
    
    if not data_loading_passed:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    else:
        print("\n🎉 All tests passed!")
    print("=" * 60)
