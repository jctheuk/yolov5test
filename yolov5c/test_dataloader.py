#!/usr/bin/env python3
"""
Test script to check if the dataloader is loading labels correctly
"""

import sys
import torch
from pathlib import Path

# Add the yolov5c directory to the path
sys.path.append(str(Path(__file__).parent))

from utils.dataloaders import create_dataloader
from utils.general import check_dataset

def test_dataloader():
    print("Testing dataloader...")
    
    # Load dataset configuration
    data_yaml = "../Regurgitation-YOLODataset-Detection/data.yaml"
    data_dict = check_dataset(data_yaml)
    
    print(f"Dataset config: {data_dict}")
    
    # Create dataloader
    train_loader = create_dataloader(
        path=data_dict['train'],
        imgsz=416,
        batch_size=2,
        stride=32,
        hyp=None,  # Use default hyperparameters
        augment=False,  # No augmentation for testing
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        workers=0,  # No multiprocessing for debugging
        image_weights=False,
        quad=False,
        prefix='',
        shuffle=False,  # No shuffling for consistent debugging
        seed=0
    )
    
    print(f"Created dataloader with {len(train_loader)} batches")
    
    # Test first few batches
    for batch_idx, (imgs, targets, classification_labels, paths, shapes) in enumerate(train_loader):
        print(f"\n=== Batch {batch_idx} ===")
        print(f"Images shape: {imgs.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Targets content: {targets}")
        print(f"Classification labels shape: {classification_labels.shape}")
        print(f"Classification labels content: {classification_labels}")
        print(f"Paths: {paths}")
        
        # Check if targets have valid values
        if targets.numel() > 0:
            print(f"Targets min: {targets.min()}, max: {targets.max()}")
            print(f"Targets mean: {targets.mean()}")
        
        # Check classification labels
        if classification_labels.numel() > 0:
            print(f"Classification labels min: {classification_labels.min()}, max: {classification_labels.max()}")
            print(f"Classification labels mean: {classification_labels.mean()}")
        
        # Only test first 3 batches
        if batch_idx >= 2:
            break
    
    print("\nDataloader test completed!")

if __name__ == "__main__":
    test_dataloader() 