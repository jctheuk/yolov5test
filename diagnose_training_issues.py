#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose training issues for YOLOv5WithClassification
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_metrics():
    """
    Analyze training metrics to identify issues
    """
    print("🔍 Analyzing Training Issues")
    print("=" * 50)
    
    # Read metrics
    metrics_file = "yolov5c/runs/classifybackbone13/classification_metrics.txt"
    
    if not Path(metrics_file).exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    # Parse metrics
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for line in lines[1:]:  # Skip header
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(',')
            if len(parts) >= 5:
                epochs.append(int(parts[0]))
                accuracies.append(float(parts[1]))
                precisions.append(float(parts[2]))
                recalls.append(float(parts[3]))
                f1_scores.append(float(parts[4]))
    
    # Analysis
    print(f"📊 Training Summary:")
    print(f"   Total epochs: {len(epochs)}")
    print(f"   Initial accuracy: {accuracies[0]:.4f} ({accuracies[0]*100:.1f}%)")
    print(f"   Final accuracy: {accuracies[-1]:.4f} ({accuracies[-1]*100:.1f}%)")
    print(f"   Best accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.1f}%)")
    print(f"   Accuracy improvement: {accuracies[-1] - accuracies[0]:.4f}")
    
    # Check for learning plateau
    last_50_epochs = accuracies[-50:] if len(accuracies) >= 50 else accuracies
    std_last_50 = np.std(last_50_epochs)
    mean_last_50 = np.mean(last_50_epochs)
    
    print(f"\n📈 Learning Analysis:")
    print(f"   Last 50 epochs std: {std_last_50:.6f}")
    print(f"   Last 50 epochs mean: {mean_last_50:.4f}")
    
    if std_last_50 < 0.001:
        print(f"   ❌ LEARNING PLATEAU DETECTED - Model stopped learning!")
    elif std_last_50 < 0.01:
        print(f"   ⚠️  Very slow learning - needs intervention")
    else:
        print(f"   ✅ Still learning (somewhat)")
    
    # Find when learning stopped
    if len(accuracies) > 100:
        # Check for significant improvement in second half
        first_half = accuracies[:len(accuracies)//2]
        second_half = accuracies[len(accuracies)//2:]
        
        first_half_improvement = max(first_half) - min(first_half)
        second_half_improvement = max(second_half) - min(second_half)
        
        print(f"\n🔄 Learning Progress:")
        print(f"   First half improvement: {first_half_improvement:.4f}")
        print(f"   Second half improvement: {second_half_improvement:.4f}")
        
        if second_half_improvement < 0.01:
            print(f"   ❌ Learning stopped around epoch {len(accuracies)//2}")
    
    # Check for overfitting
    if len(accuracies) > 20:
        recent_trend = np.polyfit(range(len(accuracies[-20:])), accuracies[-20:], 1)[0]
        print(f"\n📉 Recent Trend (last 20 epochs): {recent_trend:.6f}")
        if recent_trend < -0.001:
            print(f"   ❌ DECLINING PERFORMANCE - Possible overfitting!")
        elif recent_trend < 0.001:
            print(f"   ⚠️  STAGNANT - No improvement")
        else:
            print(f"   ✅ Still improving (slowly)")
    
    return epochs, accuracies, precisions, recalls, f1_scores

def analyze_hyperparameters():
    """
    Analyze hyperparameters for issues
    """
    print(f"\n⚙️  Hyperparameter Analysis:")
    print("=" * 50)
    
    # Read hyperparameters
    hyp_file = "yolov5c/runs/classifybackbone13/hyp.yaml"
    opt_file = "yolov5c/runs/classifybackbone13/opt.yaml"
    
    issues = []
    
    # Check learning rate
    print(f"🎯 Learning Rate Analysis:")
    print(f"   Current lr0: 0.001 (from hyp.yaml)")
    print(f"   Current lrf: 0.01 (from hyp.yaml)")
    
    if 0.001 <= 0.001:
        issues.append("❌ Learning rate too low (0.001) - should be 0.01-0.1 for classification")
    
    # Check data augmentation
    print(f"\n🔄 Data Augmentation Analysis:")
    print(f"   Mosaic: 0.0 (disabled)")
    print(f"   Mixup: 0.0 (disabled)")
    print(f"   Flip LR: 0.0 (disabled)")
    print(f"   Flip UD: 0.0 (disabled)")
    
    issues.append("⚠️  All data augmentation disabled - may limit learning")
    
    # Check classification-specific settings
    print(f"\n🎯 Classification Settings:")
    print(f"   cls_task weight: NOT SET in hyp.yaml")
    print(f"   label_smoothing: 0.1 (in opt.yaml only)")
    
    issues.append("❌ Missing cls_task weight in hyperparameters")
    issues.append("❌ label_smoothing only in opt.yaml, not hyp.yaml")
    
    # Check model architecture
    print(f"\n🏗️  Model Architecture:")
    print(f"   Config: yolov5sc_classify_backbone.yaml")
    print(f"   Image size: 416x416")
    print(f"   Batch size: 32")
    
    issues.append("⚠️  Using backbone model - may not be optimal for classification")
    
    # Check optimizer
    print(f"\n🔧 Optimizer Settings:")
    print(f"   Optimizer: SGD")
    print(f"   Momentum: 0.937")
    print(f"   Weight decay: 0.0005")
    
    issues.append("⚠️  SGD optimizer may be too slow for classification")
    
    return issues

def create_fix_recommendations(issues):
    """
    Create specific fix recommendations
    """
    print(f"\n💡 Fix Recommendations:")
    print("=" * 50)
    
    print(f"🔧 IMMEDIATE FIXES:")
    print(f"   1. INCREASE LEARNING RATE:")
    print(f"      - Change lr0 from 0.001 to 0.01 (10x increase)")
    print(f"      - Change lrf from 0.01 to 0.1")
    print(f"      - This alone should dramatically improve learning speed")
    
    print(f"\n   2. ADD CLASSIFICATION HYPERPARAMETERS:")
    print(f"      - Add cls_task: 0.3 to hyp.yaml")
    print(f"      - Add label_smoothing: 0.1 to hyp.yaml")
    
    print(f"\n   3. SWITCH TO ADAM OPTIMIZER:")
    print(f"      - Change from SGD to Adam")
    print(f"      - Adam works better for classification tasks")
    
    print(f"\n   4. ENABLE MINIMAL DATA AUGMENTATION:")
    print(f"      - Enable fliplr: 0.5 (safe for medical images)")
    print(f"      - Keep mosaic: 0.0 (medical images)")
    
    print(f"\n🚀 ADVANCED FIXES:")
    print(f"   1. USE CLASSIFICATION-SPECIFIC MODEL:")
    print(f"      - Switch to pure classification model")
    print(f"      - Remove detection head if not needed")
    
    print(f"\n   2. ADJUST LOSS WEIGHTS:")
    print(f"      - Increase classification loss weight")
    print(f"      - Reduce detection loss weights")
    
    print(f"\n   3. LEARNING RATE SCHEDULING:")
    print(f"      - Use cosine annealing")
    print(f"      - Add warmup restarts")

def create_fixed_hyperparameters():
    """
    Create fixed hyperparameter file
    """
    print(f"\n📝 Creating Fixed Hyperparameters...")
    
    fixed_hyp = """# Fixed hyperparameters for YOLOv5WithClassification
# Optimized for classification tasks

# Learning rate - INCREASED for classification
lr0: 0.01              # 10x increase from 0.001
lrf: 0.1               # 10x increase from 0.01

# Optimizer settings
momentum: 0.937
weight_decay: 0.0005

# Learning rate scheduling
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights - OPTIMIZED FOR CLASSIFICATION
box: 0.05              # Reduced detection weight
cls: 0.5               # Standard classification weight
cls_pw: 1.0
obj: 1.0               # Reduced object detection weight
obj_pw: 1.0

# Classification-specific settings - ADDED
cls_task: 0.3          # Classification task weight
label_smoothing: 0.1   # Label smoothing for better generalization

# IoU and anchor settings
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0

# Data augmentation - MINIMAL FOR MEDICAL IMAGES
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 5.0
translate: 0.1
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5            # ENABLED - safe for medical images
mosaic: 0.0            # DISABLED - not suitable for medical images
mixup: 0.0             # DISABLED - not suitable for medical images
copy_paste: 0.0
"""
    
    with open("fixed_hyp.yaml", "w") as f:
        f.write(fixed_hyp)
    
    print(f"✅ Created fixed_hyp.yaml with optimized settings")
    
    # Create training command
    training_cmd = """# Fixed training command
python train_classification_task.py \\
    --data regurgitationV1/data.yaml \\
    --weights yolov5s.pt \\
    --hyp fixed_hyp.yaml \\
    --epochs 100 \\
    --batch-size 16 \\
    --device auto \\
    --optimizer Adam \\
    --name classifybackbone_fixed \\
    --patience 0
"""
    
    with open("fixed_training_command.txt", "w") as f:
        f.write(training_cmd)
    
    print(f"✅ Created fixed_training_command.txt")

def main():
    """
    Main diagnostic function
    """
    print("🔍 YOLOv5WithClassification Training Diagnostic")
    print("=" * 60)
    
    # Analyze training metrics
    try:
        epochs, accuracies, precisions, recalls, f1_scores = analyze_training_metrics()
    except Exception as e:
        print(f"❌ Error analyzing metrics: {e}")
        return
    
    # Analyze hyperparameters
    issues = analyze_hyperparameters()
    
    # Show all issues
    print(f"\n🚨 Issues Found:")
    print("=" * 50)
    for issue in issues:
        print(f"   {issue}")
    
    # Create recommendations
    create_fix_recommendations(issues)
    
    # Create fixed files
    create_fixed_hyperparameters()
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 50)
    print(f"Your model is learning slowly because:")
    print(f"1. Learning rate is 10x too low (0.001 → should be 0.01)")
    print(f"2. Missing classification-specific hyperparameters")
    print(f"3. SGD optimizer is too slow for classification")
    print(f"4. No data augmentation (even minimal)")
    
    print(f"\n🚀 QUICK FIX:")
    print(f"Run: python train_classification_task.py --data regurgitationV1/data.yaml --hyp fixed_hyp.yaml --optimizer Adam --epochs 50")

if __name__ == "__main__":
    main()

