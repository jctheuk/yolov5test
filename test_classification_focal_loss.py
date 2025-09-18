#!/usr/bin/env python3
"""
Test script for Classification Focal Loss implementation
Tests the focal loss function with imbalanced class scenarios
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def focal_loss_classification(probs, targets, gamma=2.0, alpha=None):
    """
    Calculate Focal Loss for classification task to handle class imbalance.
    
    Args:
        probs: Softmax probabilities [batch_size, num_classes]
        targets: Target class indices [batch_size]
        gamma: Focal loss gamma parameter
        alpha: Class weights (list or float)
    
    Returns:
        Focal loss value
    """
    # Convert targets to one-hot encoding
    num_classes = probs.shape[1]
    targets_one_hot = torch.zeros_like(probs)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    
    # Calculate cross-entropy loss
    ce_loss = -torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
    ce_loss = (ce_loss * targets_one_hot).sum(dim=1)
    
    # Calculate p_t (probability of true class)
    p_t = (probs * targets_one_hot).sum(dim=1)
    
    # Calculate alpha weights
    if alpha is not None:
        if isinstance(alpha, list):
            alpha_t = torch.tensor(alpha, device=probs.device)[targets]
        else:
            alpha_t = alpha
    else:
        alpha_t = 1.0
    
    # Calculate modulating factor
    modulating_factor = (1.0 - p_t) ** gamma
    
    # Calculate focal loss
    focal_loss = alpha_t * modulating_factor * ce_loss
    
    return focal_loss.mean()

def test_focal_loss():
    """Test focal loss with imbalanced classification scenario"""
    
    print("=== Testing Classification Focal Loss ===")
    
    # Simulate imbalanced classification scenario
    # Class 0: 70% of samples (majority class)
    # Class 1: 20% of samples 
    # Class 2: 10% of samples (minority class)
    
    batch_size = 1000
    num_classes = 3
    
    # Create imbalanced targets
    targets = torch.cat([
        torch.zeros(700, dtype=torch.long),  # Class 0: 700 samples
        torch.ones(200, dtype=torch.long),   # Class 1: 200 samples
        torch.full((100,), 2, dtype=torch.long)  # Class 2: 100 samples
    ])
    
    print(f"Target distribution:")
    for i in range(num_classes):
        count = (targets == i).sum().item()
        print(f"  Class {i}: {count} samples ({count/batch_size*100:.1f}%)")
    
    # Test 1: Model predicting only majority class (overfitting scenario)
    print("\n--- Test 1: Model predicting only majority class ---")
    probs_majority = torch.tensor([
        [0.9, 0.05, 0.05],  # Always predict class 0
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05]
    ]).repeat(batch_size, 1)
    
    # Standard Cross-Entropy Loss
    ce_loss = nn.CrossEntropyLoss()
    ce_result = ce_loss(torch.log(probs_majority + 1e-8), targets)
    print(f"Standard CE Loss: {ce_result.item():.4f}")
    
    # Focal Loss
    focal_result = focal_loss_classification(probs_majority, targets, gamma=2.0)
    print(f"Focal Loss (γ=2.0): {focal_result.item():.4f}")
    
    # Test 2: Model with good predictions
    print("\n--- Test 2: Model with good predictions ---")
    probs_good = torch.tensor([
        [0.8, 0.1, 0.1],   # Good prediction for class 0
        [0.1, 0.8, 0.1],   # Good prediction for class 1
        [0.1, 0.1, 0.8]    # Good prediction for class 2
    ]).repeat(batch_size, 1)
    
    ce_result_good = ce_loss(torch.log(probs_good + 1e-8), targets)
    focal_result_good = focal_loss_classification(probs_good, targets, gamma=2.0)
    
    print(f"Standard CE Loss: {ce_result_good.item():.4f}")
    print(f"Focal Loss (γ=2.0): {focal_result_good.item():.4f}")
    
    # Test 3: Different gamma values
    print("\n--- Test 3: Different gamma values ---")
    gammas = [0.0, 1.0, 2.0, 3.0]
    for gamma in gammas:
        focal_result = focal_loss_classification(probs_majority, targets, gamma=gamma)
        print(f"Focal Loss (γ={gamma}): {focal_result.item():.4f}")
    
    # Test 4: With class weights
    print("\n--- Test 4: With class weights (alpha) ---")
    alpha_weights = [0.33, 0.33, 0.34]  # Balanced weights
    focal_result_weighted = focal_loss_classification(probs_majority, targets, gamma=2.0, alpha=alpha_weights)
    print(f"Focal Loss with weights: {focal_result_weighted.item():.4f}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_focal_loss()
