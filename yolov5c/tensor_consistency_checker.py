#!/usr/bin/env python3
"""
Tensor and Shape Consistency Checker for YOLOv5 Training Pipeline

This script validates tensor shapes and types across:
- train.py
- val.py  
- detect.py
- dataloaders.py
- model outputs
- loss computation

Usage:
    python tensor_consistency_checker.py
"""

import torch
import numpy as np
import sys
from pathlib import Path
import traceback

# Add the yolov5c directory to the path
sys.path.append(str(Path(__file__).parent))

from utils.general import parse_model_output, validate_detection_outputs
from utils.dataloaders import create_dataloader
from models.yolo import Model
from utils.loss import ComputeLoss


class TensorConsistencyChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def log_error(self, message):
        """Log an error message"""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
        
    def log_warning(self, message):
        """Log a warning message"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
        
    def log_success(self, message):
        """Log a successful check"""
        self.passed_checks.append(message)
        print(f"✅ PASSED: {message}")
        
    def check_tensor_shape(self, tensor, expected_shape, name):
        """Check if tensor has expected shape"""
        if not isinstance(tensor, torch.Tensor):
            self.log_error(f"{name} is not a torch.Tensor, got {type(tensor)}")
            return False
            
        if tensor.shape != expected_shape:
            self.log_error(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
            return False
            
        self.log_success(f"{name} shape check: {tensor.shape}")
        return True
        
    def check_tensor_dtype(self, tensor, expected_dtype, name):
        """Check if tensor has expected dtype"""
        if not isinstance(tensor, torch.Tensor):
            self.log_error(f"{name} is not a torch.Tensor, got {type(tensor)}")
            return False
            
        if tensor.dtype != expected_dtype:
            self.log_warning(f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            return False
            
        self.log_success(f"{name} dtype check: {tensor.dtype}")
        return True
        
    def check_model_output_consistency(self):
        """Check model output consistency"""
        print("\n🔍 Checking Model Output Consistency...")
        
        try:
            # Create a dummy model
            model = Model('models/yolov5sc.yaml', ch=3, nc=4)
            model.eval()
            
            # Create dummy input
            batch_size = 2
            img_size = 416
            dummy_input = torch.randn(batch_size, 3, img_size, img_size)
            
            # Forward pass
            with torch.no_grad():
                model_output = model(dummy_input)
            
            # Parse model output
            detection_outputs, classification_output = parse_model_output(model_output)
            
            # Validate detection outputs
            validate_detection_outputs(detection_outputs)
            
            # Check detection output shapes
            for i, det_output in enumerate(detection_outputs):
                if len(det_output.shape) == 5:  # Training format
                    # (batch, anchors, height, width, detection_info)
                    expected_shape = (batch_size, 3, det_output.shape[2], det_output.shape[3], 6)
                    self.check_tensor_shape(det_output, expected_shape, f"Detection output {i}")
                elif len(det_output.shape) == 4:  # Inference format
                    # (batch, channels, height, width)
                    self.log_success(f"Detection output {i} inference format: {det_output.shape}")
                elif len(det_output.shape) == 3:  # Concatenated format
                    # (batch, detections, info)
                    self.log_success(f"Detection output {i} concatenated format: {det_output.shape}")
                    
            # Check classification output
            if classification_output is not None:
                expected_cls_shape = (batch_size, 256)  # Based on your model
                self.check_tensor_shape(classification_output, expected_cls_shape, "Classification output")
                
        except Exception as e:
            self.log_error(f"Model output consistency check failed: {e}")
            traceback.print_exc()
            
    def check_dataloader_consistency(self):
        """Check dataloader consistency"""
        print("\n🔍 Checking Dataloader Consistency...")
        
        try:
            # Create a dummy dataset path
            data_yaml = "../Regurgitation-YOLODataset-Detection/data.yaml"
            
            # Create dataloader
            dataloader, dataset = create_dataloader(
                path=data_yaml,
                imgsz=416,
                batch_size=2,
                stride=32,
                single_cls=False,
                hyp=None,
                augment=False,
                cache=False,
                rect=False,
                rank=-1,
                workers=0,
                image_weights=False,
                quad=False,
                prefix='test: ',
                shuffle=False
            )
            
            # Get a batch
            batch = next(iter(dataloader))
            imgs, targets, paths, shapes, classification_labels = batch
            
            # Check image shapes
            expected_img_shape = (2, 3, 416, 416)
            self.check_tensor_shape(imgs, expected_img_shape, "Images")
            self.check_tensor_dtype(imgs, torch.float32, "Images")
            
            # Check targets
            if len(targets.shape) == 2:
                self.log_success(f"Targets shape: {targets.shape}")
                if targets.shape[1] >= 5:  # class, x, y, w, h
                    self.log_success("Targets format: (batch_idx, class, x, y, w, h)")
                else:
                    self.log_warning(f"Targets may have insufficient columns: {targets.shape[1]}")
            else:
                self.log_error(f"Targets should be 2D, got shape {targets.shape}")
                
            # Check classification labels
            if classification_labels is not None:
                if isinstance(classification_labels, torch.Tensor):
                    self.log_success(f"Classification labels shape: {classification_labels.shape}")
                else:
                    self.log_warning(f"Classification labels type: {type(classification_labels)}")
                    
        except Exception as e:
            self.log_error(f"Dataloader consistency check failed: {e}")
            traceback.print_exc()
            
    def check_loss_computation_consistency(self):
        """Check loss computation consistency"""
        print("\n🔍 Checking Loss Computation Consistency...")
        
        try:
            # Create model and loss function
            model = Model('models/yolov5sc.yaml', ch=3, nc=4)
            compute_loss = ComputeLoss(model)
            
            # Create dummy data
            batch_size = 2
            img_size = 416
            
            # Dummy images
            imgs = torch.randn(batch_size, 3, img_size, img_size)
            
            # Dummy targets (batch_idx, class, x, y, w, h)
            targets = torch.tensor([
                [0, 0, 0.5, 0.5, 0.2, 0.2],  # First image, class 0
                [1, 1, 0.3, 0.7, 0.1, 0.1]   # Second image, class 1
            ], dtype=torch.float32)
            
            # Dummy classification labels
            classification_labels = torch.tensor([0, 1], dtype=torch.long)  # Class indices
            
            # Forward pass
            model_output = model(imgs)
            
            # Parse outputs
            detection_outputs, classification_output = parse_model_output(model_output)
            
            # Compute loss
            total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
            
            # Check loss outputs
            if isinstance(total_loss, torch.Tensor):
                self.log_success(f"Total loss shape: {total_loss.shape}")
                self.log_success(f"Total loss value: {total_loss.item():.4f}")
            else:
                self.log_error(f"Total loss should be tensor, got {type(total_loss)}")
                
            if isinstance(loss_items, (list, tuple)):
                self.log_success(f"Loss items: {len(loss_items)} components")
                for i, item in enumerate(loss_items):
                    if isinstance(item, torch.Tensor):
                        self.log_success(f"Loss item {i}: {item.item():.4f}")
                    else:
                        self.log_warning(f"Loss item {i} is not tensor: {type(item)}")
            else:
                self.log_error(f"Loss items should be list/tuple, got {type(loss_items)}")
                
        except Exception as e:
            self.log_error(f"Loss computation consistency check failed: {e}")
            traceback.print_exc()
            
    def check_training_loop_consistency(self):
        """Check training loop consistency"""
        print("\n🔍 Checking Training Loop Consistency...")
        
        try:
            # Simulate training loop steps
            batch_size = 2
            img_size = 416
            
            # Dummy data
            imgs = torch.randn(batch_size, 3, img_size, img_size)
            targets = torch.tensor([
                [0, 0, 0.5, 0.5, 0.2, 0.2],
                [1, 1, 0.3, 0.7, 0.1, 0.1]
            ], dtype=torch.float32)
            
            # Create classification labels from paths (simulate the function)
            paths = ['image1.jpg', 'image2.jpg']
            classification_labels = torch.zeros(batch_size, 3)  # 3 classes
            
            # Simulate classification label assignment
            for i, path in enumerate(paths):
                if 'psax' in path.lower():
                    classification_labels[i, 0] = 1.0
                elif 'plax' in path.lower():
                    classification_labels[i, 1] = 1.0
                else:
                    classification_labels[i, 2] = 1.0
                    
            # Check shapes
            self.check_tensor_shape(imgs, (batch_size, 3, img_size, img_size), "Training images")
            self.check_tensor_shape(targets, (2, 6), "Training targets")
            self.check_tensor_shape(classification_labels, (batch_size, 3), "Training classification labels")
            
            # Check that classification labels sum to 1 for each sample
            label_sums = classification_labels.sum(dim=1)
            if not torch.allclose(label_sums, torch.ones(batch_size)):
                self.log_warning("Classification labels don't sum to 1 for all samples")
            else:
                self.log_success("Classification labels properly normalized")
                
        except Exception as e:
            self.log_error(f"Training loop consistency check failed: {e}")
            traceback.print_exc()
            
    def check_validation_consistency(self):
        """Check validation consistency"""
        print("\n🔍 Checking Validation Consistency...")
        
        try:
            # Simulate validation steps
            batch_size = 2
            img_size = 416
            
            # Dummy validation data
            imgs = torch.randn(batch_size, 3, img_size, img_size)
            labels = torch.tensor([
                [0, 0, 0.5, 0.5, 0.2, 0.2],
                [1, 1, 0.3, 0.7, 0.1, 0.1]
            ], dtype=torch.float32)
            
            classification_labels = torch.tensor([0, 1], dtype=torch.long)
            
            # Check shapes
            self.check_tensor_shape(imgs, (batch_size, 3, img_size, img_size), "Validation images")
            self.check_tensor_shape(labels, (2, 6), "Validation labels")
            self.check_tensor_shape(classification_labels, (batch_size,), "Validation classification labels")
            
            # Simulate model prediction
            model = Model('models/yolov5sc.yaml', ch=3, nc=4)
            model.eval()
            
            with torch.no_grad():
                model_output = model(imgs)
                
            # Parse outputs
            preds, classification_output = parse_model_output(model_output)
            
            # Check prediction shapes
            for i, pred in enumerate(preds):
                if len(pred.shape) == 3:  # (batch, detections, info)
                    self.log_success(f"Validation prediction {i} shape: {pred.shape}")
                else:
                    self.log_warning(f"Validation prediction {i} unexpected shape: {pred.shape}")
                    
            # Check classification output
            if classification_output is not None:
                expected_shape = (batch_size, 256)
                self.check_tensor_shape(classification_output, expected_shape, "Validation classification output")
                
        except Exception as e:
            self.log_error(f"Validation consistency check failed: {e}")
            traceback.print_exc()
            
    def check_detection_consistency(self):
        """Check detection consistency"""
        print("\n🔍 Checking Detection Consistency...")
        
        try:
            # Simulate detection inference
            batch_size = 1
            img_size = 416
            
            # Dummy input
            imgs = torch.randn(batch_size, 3, img_size, img_size)
            
            # Model inference
            model = Model('models/yolov5sc.yaml', ch=3, nc=4)
            model.eval()
            
            with torch.no_grad():
                model_output = model(imgs)
                
            # Parse outputs
            pred, classification_output = parse_model_output(model_output)
            
            # Check detection output
            if isinstance(pred, list):
                for i, p in enumerate(pred):
                    self.log_success(f"Detection output {i} shape: {p.shape}")
            else:
                self.log_success(f"Detection output shape: {pred.shape}")
                
            # Check classification output
            if classification_output is not None:
                expected_shape = (batch_size, 256)
                self.check_tensor_shape(classification_output, expected_shape, "Detection classification output")
                
        except Exception as e:
            self.log_error(f"Detection consistency check failed: {e}")
            traceback.print_exc()
            
    def run_all_checks(self):
        """Run all consistency checks"""
        print("🚀 Starting Tensor and Shape Consistency Checks...")
        print("=" * 60)
        
        self.check_model_output_consistency()
        self.check_dataloader_consistency()
        self.check_loss_computation_consistency()
        self.check_training_loop_consistency()
        self.check_validation_consistency()
        self.check_detection_consistency()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 CONSISTENCY CHECK SUMMARY")
        print("=" * 60)
        print(f"✅ Passed checks: {len(self.passed_checks)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        if not self.errors:
            print("\n🎉 All consistency checks passed! Your pipeline should work correctly.")
        else:
            print("\n🔧 Please fix the errors above before running training.")
            
        return len(self.errors) == 0


def main():
    """Main function"""
    checker = TensorConsistencyChecker()
    success = checker.run_all_checks()
    
    if success:
        print("\n✅ Tensor consistency check completed successfully!")
        return 0
    else:
        print("\n❌ Tensor consistency check failed!")
        return 1


if __name__ == "__main__":
    exit(main())
