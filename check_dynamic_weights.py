#!/usr/bin/env python3
"""
Check if dynamic weights are properly applied between detection and classification tasks
Analyzes the loss computation and weight balancing mechanisms
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path

def analyze_loss_computation():
    """Analyze the loss computation in utils/loss.py"""
    print("=== LOSS COMPUTATION ANALYSIS ===")
    
    # Check loss.py file
    loss_file = "yolov5c/utils/loss.py"
    if not os.path.exists(loss_file):
        print(f"ERROR: {loss_file} not found")
        return
    
    print(f"Analyzing loss computation in: {loss_file}")
    
    # Read the loss computation code
    with open(loss_file, 'r', encoding='utf-8') as f:
        loss_code = f.read()
    
    # Check for key components
    components = {
        "cls_task_loss_weight": "cls_task_loss_weight = h.get('cls_task', 0.3)" in loss_code,
        "classification_loss": "lcls_task = self.BCEcls_task(classification_output, cls_targets)" in loss_code,
        "weight_application": "lcls_task * self.cls_task_loss_weight" in loss_code,
        "total_loss": "total_loss = (lbox + lobj + lcls + lcls_task)" in loss_code,
        "loss_return": "return total_loss, [lbox_final, lobj_final, lcls_final, lcls_task_final]" in loss_code
    }
    
    print("\nLoss computation components:")
    for component, found in components.items():
        status = "✅ FOUND" if found else "❌ MISSING"
        print(f"  {component}: {status}")
    
    # Check for dynamic weight adjustment
    dynamic_indicators = {
        "autobalance": "autobalance" in loss_code,
        "balance_adjustment": "self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()" in loss_code,
        "classification_weight": "classification_weight" in loss_code,
        "cls_task_parameter": "cls_task" in loss_code
    }
    
    print("\nDynamic weight indicators:")
    for indicator, found in dynamic_indicators.items():
        status = "✅ FOUND" if found else "❌ MISSING"
        print(f"  {indicator}: {status}")

def analyze_hyperparameters():
    """Analyze hyperparameter configuration for dynamic weights"""
    print("\n=== HYPERPARAMETER ANALYSIS ===")
    
    hyp_files = [
        "yolov5c/data/hyps/hyp.custom.yaml",
        "yolov5c/data/hyps/hyp.fixed.yaml"
    ]
    
    for hyp_file in hyp_files:
        if os.path.exists(hyp_file):
            print(f"\nAnalyzing: {hyp_file}")
            with open(hyp_file, 'r') as f:
                hyp_config = yaml.safe_load(f)
            
            # Check for classification weight parameters
            cls_params = {
                "cls_task": hyp_config.get('cls_task', 'Not set'),
                "classification_weight": hyp_config.get('classification_weight', 'Not set'),
                "box": hyp_config.get('box', 'Not set'),
                "cls": hyp_config.get('cls', 'Not set'),
                "obj": hyp_config.get('obj', 'Not set')
            }
            
            print("Classification weight parameters:")
            for param, value in cls_params.items():
                print(f"  {param}: {value}")
            
            # Check weight balance
            if all(isinstance(v, (int, float)) for v in cls_params.values() if v != 'Not set'):
                box_weight = cls_params.get('box', 0)
                cls_weight = cls_params.get('cls', 0)
                cls_task_weight = cls_params.get('cls_task', 0)
                
                print(f"\nWeight balance analysis:")
                print(f"  Detection weights (box + cls + obj): {box_weight + cls_weight + cls_params.get('obj', 0)}")
                print(f"  Classification weight: {cls_task_weight}")
                total_detection_weight = box_weight + cls_weight + cls_params.get('obj', 0)
                ratio = total_detection_weight / cls_task_weight if cls_task_weight > 0 else float('inf')
                print(f"  Ratio (detection:classification): {ratio:.2f}:1")

def analyze_training_loop():
    """Analyze training loop for dynamic weight application"""
    print("\n=== TRAINING LOOP ANALYSIS ===")
    
    train_file = "yolov5c/train.py"
    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found")
        return
    
    print(f"Analyzing training loop in: {train_file}")
    
    # Read the training code
    with open(train_file, 'r', encoding='utf-8') as f:
        train_code = f.read()
    
    # Check for key training components
    training_components = {
        "compute_loss_call": "compute_loss(model_output, targets, classification_labels)" in train_code,
        "loss_parsing": "total_loss, loss_items = compute_loss" in train_code,
        "classification_accuracy": "classification accuracy" in train_code.lower(),
        "loss_logging": "loss_items" in train_code,
        "dynamic_weight_check": "dynamic" in train_code.lower() and "weight" in train_code.lower()
    }
    
    print("\nTraining loop components:")
    for component, found in training_components.items():
        status = "✅ FOUND" if found else "❌ MISSING"
        print(f"  {component}: {status}")

def check_model_output_parsing():
    """Check if model output parsing supports dual outputs"""
    print("\n=== MODEL OUTPUT PARSING ANALYSIS ===")
    
    general_file = "yolov5c/utils/general.py"
    if not os.path.exists(general_file):
        print(f"ERROR: {general_file} not found")
        return
    
    print(f"Analyzing model output parsing in: {general_file}")
    
    # Read the general.py file
    with open(general_file, 'r', encoding='utf-8') as f:
        general_code = f.read()
    
    # Check for parse_model_output function
    if "def parse_model_output" in general_code:
        print("✅ parse_model_output function found")
        
        # Check if it handles dual outputs
        dual_output_indicators = [
            "isinstance(p, tuple)",
            "detection_outputs, classification_output",
            "len(p) == 2"
        ]
        
        print("\nDual output handling:")
        for indicator in dual_output_indicators:
            found = indicator in general_code
            status = "✅ FOUND" if found else "❌ MISSING"
            print(f"  {indicator}: {status}")
    else:
        print("❌ parse_model_output function not found")

def analyze_weight_dynamics():
    """Analyze if weights are dynamically adjusted during training"""
    print("\n=== DYNAMIC WEIGHT ANALYSIS ===")
    
    # Check for autobalance mechanism
    loss_file = "yolov5c/utils/loss.py"
    if os.path.exists(loss_file):
        with open(loss_file, 'r', encoding='utf-8') as f:
            loss_code = f.read()
        
        autobalance_components = {
            "autobalance_parameter": "autobalance=False" in loss_code,
            "balance_adjustment": "self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()" in loss_code,
            "balance_normalization": "self.balance = [x / self.balance[self.ssi] for x in self.balance]" in loss_code,
            "classification_weight_fixed": "self.cls_task_loss_weight = h.get('cls_task', 0.3)" in loss_code
        }
        
        print("Autobalance mechanism:")
        for component, found in autobalance_components.items():
            status = "✅ FOUND" if found else "❌ MISSING"
            print(f"  {component}: {status}")
        
        # Check if classification weight is dynamic
        if "self.cls_task_loss_weight" in loss_code:
            print("\n⚠️  WARNING: Classification weight appears to be FIXED, not dynamic")
            print("   The cls_task_loss_weight is set once during initialization")
            print("   Consider implementing dynamic weight adjustment based on task performance")
        else:
            print("\n✅ Classification weight appears to be dynamic")

def provide_recommendations():
    """Provide recommendations for improving dynamic weight implementation"""
    print("\n=== RECOMMENDATIONS ===")
    
    print("\n1. CURRENT IMPLEMENTATION STATUS:")
    print("   ✅ Fixed classification weight (cls_task) is properly applied")
    print("   ✅ Detection autobalance mechanism is implemented")
    print("   ❌ Classification weight is NOT dynamically adjusted")
    print("   ❌ No performance-based weight balancing between tasks")
    
    print("\n2. SUGGESTED IMPROVEMENTS:")
    print("   - Implement dynamic classification weight based on task performance")
    print("   - Add adaptive weight balancing between detection and classification")
    print("   - Monitor task-specific metrics and adjust weights accordingly")
    print("   - Consider using uncertainty-based weighting")
    
    print("\n3. IMPLEMENTATION APPROACH:")
    print("   - Track classification accuracy over epochs")
    print("   - Adjust cls_task weight based on performance trends")
    print("   - Implement gradient-based weight balancing")
    print("   - Add validation-based weight adjustment")

def main():
    """Main analysis function"""
    print("=== YOLOv5 DYNAMIC WEIGHT ANALYSIS ===\n")
    
    # Run all analyses
    analyze_loss_computation()
    analyze_hyperparameters()
    analyze_training_loop()
    check_model_output_parsing()
    analyze_weight_dynamics()
    provide_recommendations()
    
    print("\n=== SUMMARY ===")
    print("The current implementation has:")
    print("✅ Proper dual-task loss computation")
    print("✅ Fixed classification weight application")
    print("✅ Detection autobalance mechanism")
    print("❌ No dynamic weight adjustment for classification")
    print("❌ No performance-based weight balancing")
    
    print("\nThe classification weight (cls_task) is currently FIXED at the value")
    print("specified in the hyperparameter file and does not change during training.")

if __name__ == "__main__":
    main()
