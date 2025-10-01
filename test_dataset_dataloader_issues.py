#!/usr/bin/env python3
"""
Dataset and Dataloader Diagnostic Script
Following CLASSIFICATION_ISSUES_TODO_LIST.md to identify root cause
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import cv2
from collections import Counter
import matplotlib.pyplot as plt

# Add yolov5c to path
sys.path.append('yolov5c')

def test_dataset_structure():
    """Test 1: Check dataset structure and label format"""
    print("=" * 60)
    print("🔍 TEST 1: Dataset Structure & Label Format")
    print("=" * 60)
    
    # Check data.yaml
    data_yaml = "regurgitationV1/data.yaml"
    print(f"📁 Checking {data_yaml}...")
    
    if not Path(data_yaml).exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("✅ Data configuration:")
    print(f"   Detection classes: {data_config.get('nc', 'NOT FOUND')}")
    print(f"   Detection names: {data_config.get('names', 'NOT FOUND')}")
    print(f"   Classification classes: {data_config.get('num_cls', 'NOT FOUND')}")
    print(f"   Classification names: {data_config.get('cls_names', 'NOT FOUND')}")
    
    # Check if classification config exists
    if 'num_cls' not in data_config:
        print("❌ Missing 'num_cls' in data.yaml")
        return False
    if 'cls_names' not in data_config:
        print("❌ Missing 'cls_names' in data.yaml")
        return False
    
    return True

def test_label_file_format():
    """Test 2: Validate classification label files"""
    print("\n" + "=" * 60)
    print("🔍 TEST 2: Label File Format Validation")
    print("=" * 60)
    
    label_dir = Path("regurgitationV1/train/labels")
    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return False
    
    # Check first 10 label files
    label_files = list(label_dir.glob("*.txt"))[:10]
    print(f"📁 Checking {len(label_files)} label files...")
    
    classification_labels_found = 0
    detection_labels_found = 0
    malformed_files = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
            
            detection_lines = []
            classification_line = None
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 5:  # Detection line
                    detection_lines.append(parts)
                elif len(parts) == 3:  # Classification line
                    classification_line = parts
                else:
                    malformed_files += 1
                    break
            
            if detection_lines:
                detection_labels_found += 1
            if classification_line:
                classification_labels_found += 1
                print(f"   ✅ {label_file.name}: Detection={len(detection_lines)}, Classification={classification_line}")
            else:
                print(f"   ❌ {label_file.name}: No classification label found")
                
        except Exception as e:
            print(f"   ❌ Error reading {label_file.name}: {e}")
            malformed_files += 1
    
    print(f"\n📊 Summary:")
    print(f"   Files with detection labels: {detection_labels_found}/{len(label_files)}")
    print(f"   Files with classification labels: {classification_labels_found}/{len(label_files)}")
    print(f"   Malformed files: {malformed_files}")
    
    if classification_labels_found == 0:
        print("❌ NO CLASSIFICATION LABELS FOUND IN ANY FILES!")
        return False
    
    return True

def test_class_distribution():
    """Test 3: Class distribution analysis"""
    print("\n" + "=" * 60)
    print("🔍 TEST 3: Class Distribution Analysis")
    print("=" * 60)
    
    label_dir = Path("regurgitationV1/train/labels")
    class_counts = Counter()
    total_files = 0
    files_without_classification = 0
    
    print("📊 Analyzing class distribution...")
    
    for label_file in label_dir.glob("*.txt"):
        total_files += 1
        try:
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
            
            classification_line = None
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 3:  # Classification line
                    classification_line = parts
                    break
            
            if classification_line:
                # Convert one-hot to class index
                class_idx = classification_line.index('1')
                class_counts[class_idx] += 1
            else:
                files_without_classification += 1
                
        except Exception as e:
            files_without_classification += 1
    
    class_names = ['A4C', 'PSAX', 'PLAX']
    print(f"\n📈 Class Distribution (out of {total_files} files):")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"   {name} (class {i}): {count} files ({percentage:.1f}%)")
    
    print(f"   Files without classification: {files_without_classification}")
    
    # Check for imbalance
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 2:
            print(f"⚠️  Class imbalance detected: ratio {imbalance_ratio:.1f}:1")
        else:
            print("✅ Class distribution is balanced")
    
    return True

def test_dataloader_loading():
    """Test 4: Check dataloader configuration and loading"""
    print("\n" + "=" * 60)
    print("🔍 TEST 4: Dataloader Configuration & Loading")
    print("=" * 60)
    
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("📁 Testing dataloader creation...")
        
        # Create dataloader
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=4,
            gs=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=None,
            rect=False,
            rank=-1,
            workers=4,
            prefix='',
            shuffle=False
        )
        
        print("✅ Dataloader created successfully")
        
        # Test first batch
        print("📊 Testing first batch...")
        for i, batch in enumerate(train_loader):
            images, targets, paths, shapes, classification_labels = batch
            
            print(f"   Batch {i}:")
            print(f"     Images shape: {images.shape}")
            print(f"     Targets shape: {targets.shape}")
            print(f"     Classification labels shape: {classification_labels.shape}")
            print(f"     Classification labels dtype: {classification_labels.dtype}")
            print(f"     Classification labels: {classification_labels}")
            
            # Check if all labels are the same (default fallback)
            unique_labels = torch.unique(classification_labels, dim=0)
            print(f"     Unique classification labels: {len(unique_labels)}")
            
            if len(unique_labels) == 1:
                print("     ❌ ALL SAMPLES HAVE SAME CLASSIFICATION LABEL - DEFAULT FALLBACK DETECTED!")
                print("     This indicates classification labels are not being loaded from files!")
                return False
            else:
                print("     ✅ Multiple unique classification labels found")
            
            # Check label values
            for j, label in enumerate(classification_labels):
                class_idx = label.argmax().item()
                confidence = label.max().item()
                print(f"       Sample {j}: Class {class_idx}, Confidence {confidence:.3f}")
            
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_loading():
    """Test 5: Check cache loading"""
    print("\n" + "=" * 60)
    print("🔍 TEST 5: Cache Loading Analysis")
    print("=" * 60)
    
    cache_file = Path("regurgitationV1/train/labels.cache")
    
    if not cache_file.exists():
        print("❌ Cache file not found - this is normal for first run")
        print("   Cache will be created during first dataloader creation")
        return True
    
    print(f"📁 Checking cache file: {cache_file}")
    
    try:
        import pickle
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        print(f"✅ Cache loaded successfully")
        print(f"   Cache keys: {len(cache)} items")
        
        # Check first few cache entries
        sample_keys = list(cache.keys())[:3]
        print(f"\n📊 Sample cache entries:")
        
        for key in sample_keys:
            value = cache[key]
            print(f"   {Path(key).name}:")
            print(f"     Type: {type(value)}")
            if isinstance(value, tuple) and len(value) >= 4:
                labels, shape, segments, classification_labels = value[:4]
                print(f"     Labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                print(f"     Classification labels: {classification_labels}")
            else:
                print(f"     Value: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return False

def test_model_output():
    """Test 6: Check model output format"""
    print("\n" + "=" * 60)
    print("🔍 TEST 6: Model Output Format")
    print("=" * 60)
    
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        from yolov5c.models.experimental import attempt_load
        
        print("📁 Loading model...")
        model = attempt_load(model_path, device='cpu', inplace=True, fuse=True)
        model.eval()
        
        print("✅ Model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 416, 416)
        
        print("📊 Testing model forward pass...")
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"   Output type: {type(outputs)}")
        
        if isinstance(outputs, tuple):
            print(f"   Output length: {len(outputs)}")
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"     Output {i} shape: {output.shape}")
                else:
                    print(f"     Output {i}: {output}")
        else:
            if hasattr(outputs, 'shape'):
                print(f"   Output shape: {outputs.shape}")
            else:
                print(f"   Output: {outputs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 Dataset & Dataloader Diagnostic Suite")
    print("Following CLASSIFICATION_ISSUES_TODO_LIST.md")
    print("=" * 80)
    
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Label File Format", test_label_file_format),
        ("Class Distribution", test_class_distribution),
        ("Dataloader Loading", test_dataloader_loading),
        ("Cache Loading", test_cache_loading),
        ("Model Output", test_model_output),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dataset and dataloader appear to be working correctly.")
        print("   The issue might be in model architecture, loss function, or training configuration.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        
        # Specific recommendations
        if not results.get("Label File Format", True):
            print("\n🔧 RECOMMENDATION: Fix label file format issues first")
        if not results.get("Dataloader Loading", True):
            print("\n🔧 RECOMMENDATION: Fix dataloader classification label loading")
        if not results.get("Class Distribution", True):
            print("\n🔧 RECOMMENDATION: Address class imbalance or missing labels")

if __name__ == "__main__":
    main()

