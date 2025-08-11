#!/usr/bin/env python3
"""
Minimal test for data loading functionality
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test data loading with classification labels"""
    
    print("\nTesting data loading...")
    
    try:
        # Use the real dataset from Regurgitation-YOLODataset-Detection
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
        from utils.dataloaders import create_dataloader
        from utils.general import check_dataset
        from utils.torch_utils import select_device
        
        device = select_device('')
        
        # Test check_dataset directly
        print(f"\n🔍 Debug: Testing check_dataset directly...")
        try:
            data_dict = check_dataset(str(temp_data_yaml))
            print(f"🔍 Debug: check_dataset returned: {data_dict}")
            print(f"🔍 Debug: train path from check_dataset: {data_dict.get('train', 'Not found')}")
        except Exception as e:
            print(f"❌ check_dataset failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Use the temporary data.yaml with absolute paths
        try:
            print(f"\n🔍 Debug: Calling create_dataloader with {temp_data_yaml}")
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

if __name__ == '__main__':
    print("=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)
    
    success = test_data_loading()
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Data loading: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if not success:
        print("\n⚠️  Test failed. Please check the errors above.")
    else:
        print("\n🎉 Test passed!")
    print("=" * 60)
