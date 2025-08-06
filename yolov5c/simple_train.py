#!/usr/bin/env python3
"""
Simplified YOLOv5 training script
"""
import sys
import os
from pathlib import Path

# Add the yolov5c directory to the path
sys.path.append(str(Path(__file__).parent))

from train import main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../Regurgitation-YOLODataset-Detection/data.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img', type=int, default=416, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume from last checkpoint')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box save interval to -1')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='version of dataset artifact to use')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images for faster training (ram/disk)')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine learning rate scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--nosaug', action='store_true', help='disable saved images')
    parser.add_argument('--notest', action='store_true', help='disable test')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images for faster training (ram/disk)')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine learning rate scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--nosaug', action='store_true', help='disable saved images')
    parser.add_argument('--notest', action='store_true', help='disable test')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    
    opt = parser.parse_args()
    
    print(f"Starting training with:")
    print(f"  Data: {opt.data}")
    print(f"  Config: {opt.cfg}")
    print(f"  Epochs: {opt.epochs}")
    print(f"  Batch size: {opt.batch_size}")
    print(f"  Image size: {opt.img}")
    print(f"  Device: {opt.device}")
    
    # Run the training
    main(opt) 