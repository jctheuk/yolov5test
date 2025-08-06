#!/usr/bin/env python3
# YOLOv5 Multi-Task Training Script
# Combines detection (4 classes: AR, MR, PR, TR) and classification (3 classes: echocardiogram views)

import os
import sys
import argparse
from pathlib import Path

# Add YOLOv5 root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import train, parse_opt
from utils.general import LOGGER, colorstr

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Multi-Task Training')
    parser.add_argument('--data', type=str, default='Regurgitation-YOLODataset-Detection/data.yaml', 
                       help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='models/yolov5sc.yaml', 
                       help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.custom.yaml', 
                       help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='multi_task_exp', help='save to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience')
    
    opt = parser.parse_args()
    
    # Print configuration
    LOGGER.info(colorstr('Multi-Task Training Configuration:'))
    LOGGER.info(f'  Detection classes: 4 (AR, MR, PR, TR)')
    LOGGER.info(f'  Classification classes: 3 (Apical, Parasternal, Other views)')
    LOGGER.info(f'  Model: {opt.cfg}')
    LOGGER.info(f'  Dataset: {opt.data}')
    LOGGER.info(f'  Epochs: {opt.epochs}')
    LOGGER.info(f'  Batch size: {opt.batch_size}')
    LOGGER.info(f'  Image size: {opt.imgsz}')
    
    # Convert to train.py format
    train_opt = parse_opt()
    train_opt.data = opt.data
    train_opt.cfg = opt.cfg
    train_opt.hyp = opt.hyp
    train_opt.epochs = opt.epochs
    train_opt.batch_size = opt.batch_size
    train_opt.imgsz = opt.imgsz
    train_opt.weights = opt.weights
    train_opt.project = opt.project
    train_opt.name = opt.name
    train_opt.device = opt.device
    train_opt.workers = opt.workers
    train_opt.patience = opt.patience
    
    # Start training
    from train import main as train_main
    train_main(train_opt)

if __name__ == '__main__':
    main() 