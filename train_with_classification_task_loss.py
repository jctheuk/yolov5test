#!/usr/bin/env python3
"""
Modified training script that uses ClassificationTaskLoss instead of ComputeLoss
This script shows how to integrate the new loss function into your existing training pipeline
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent / "yolov5c"))

from yolov5c.utils.classification_task_loss import ClassificationTaskLoss
from yolov5c.models.yolo import Model
from yolov5c.utils.general import LOGGER, colorstr
from yolov5c.utils.torch_utils import select_device, smart_optimizer


def train_with_classification_task_loss(opt):
    """
    Training function using ClassificationTaskLoss
    """
    device = select_device(opt.device)
    
    # Load your model (same as before)
    model = Model(opt.cfg, ch=3, nc=opt.nc).to(device)
    
    # REPLACE ComputeLoss with ClassificationTaskLoss
    compute_loss = ClassificationTaskLoss(
        model=model,
        enable_classification=True,
        cls_task_weight=opt.cls_task_weight,  # Use command line parameter
        label_smoothing=opt.label_smoothing
    )
    
    # Optimizer (same as before)
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, opt.momentum, opt.weight_decay)
    
    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        
        # Your training loop here
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            pred = model(images)
            
            # Compute loss using ClassificationTaskLoss
            loss, loss_items = compute_loss(pred, targets.to(device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if batch_idx % 10 == 0:
                LOGGER.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={loss.item():.4f}, "
                          f"Cls_Loss={loss_items[0].item():.4f}, "
                          f"Cls_Weight={compute_loss.get_classification_weight():.3f}")


def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Existing arguments
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default="", help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="optimizer weight decay")
    
    # Classification task loss specific arguments
    parser.add_argument("--cls-task-weight", type=float, default=0.3, help="classification task loss weight")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="label smoothing epsilon")
    parser.add_argument("--nc", type=int, default=2, help="number of classes")
    
    return parser.parse_args()


def main():
    """Main function"""
    opt = parse_opt()
    
    LOGGER.info(f"Starting training with ClassificationTaskLoss")
    LOGGER.info(f"Classification task weight: {opt.cls_task_weight}")
    LOGGER.info(f"Label smoothing: {opt.label_smoothing}")
    
    train_with_classification_task_loss(opt)


if __name__ == "__main__":
    main()
