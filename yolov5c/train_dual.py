#!/usr/bin/env python3
"""
Dual-task YOLOv5 training script for detection + classification
Optimized for echocardiogram analysis
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add yolov5c to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, init_seeds, LOGGER
from utils.dual_loss import ComputeDualLoss
from utils.torch_utils import select_device, smart_optimizer
from utils.metrics import fitness
from utils.plots import plot_classification_metrics
import matplotlib.pyplot as plt

def train_dual_model(
    data_yaml='data.yaml',
    weights='yolov5s.pt',
    epochs=100,
    batch_size=16,
    imgsz=640,
    device='',
    workers=8,
    project='runs/train',
    name='dual_exp',
    exist_ok=False,
    resume=False,
    hyp_yaml='data/hyps/hyp.scratch-low.yaml'
):
    """
    Train a dual-task YOLOv5 model for detection + classification
    """
    
    # Initialize
    device = select_device(device)
    init_seeds(1)
    
    # Load hyperparameters
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Load dataset config
    data_dict = check_dataset(data_yaml)
    nc = data_dict['nc']  # number of detection classes
    num_cls = data_dict.get('num_cls', 3)  # number of classification classes
    
    # Create model
    model = Model('models/yolov5sc.yaml', ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # Load pretrained weights if specified
    if weights and weights != '':
        if os.path.isfile(weights):
            ckpt = torch.load(weights, map_location=device)
            model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
            LOGGER.info(f'Loaded pretrained weights from {weights}')
        else:
            LOGGER.warning(f'Weights file {weights} not found, training from scratch')
    
    # Create dataloaders
    train_loader, train_dataset = create_dataloader(
        path=data_dict['train'],
        imgsz=imgsz,
        batch_size=batch_size,
        stride=32,
        hyp=hyp,
        augment=True,
        cache=False,
        rect=False,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix='train: ',
        shuffle=True
    )
    
    val_loader, val_dataset = create_dataloader(
        path=data_dict['val'],
        imgsz=imgsz,
        batch_size=batch_size * 2,
        stride=32,
        hyp=hyp,
        augment=False,
        cache=False,
        rect=True,
        rank=-1,
        workers=workers * 2,
        image_weights=False,
        quad=False,
        prefix='val: ',
        shuffle=False
    )
    
    # Initialize loss function
    compute_loss = ComputeDualLoss(model)
    
    # Initialize optimizer
    optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    LOGGER.info(f'Starting training for {epochs} epochs...')
    LOGGER.info(f'Detection classes: {nc}, Classification classes: {num_cls}')
    
    best_fitness = 0.0
    train_losses = []
    val_losses = []
    classification_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls_acc = 0.0
        num_batches = 0
        
        # Training
        for batch_idx, (imgs, targets, paths, shapes, classification_labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            classification_labels = classification_labels.to(device)
            
            # Forward pass
            model_output = model(imgs)
            
            # Compute loss
            total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Calculate classification accuracy
            if isinstance(model_output, tuple) and len(model_output) == 2:
                _, cls_output = model_output
                if cls_output is not None:
                    pred_classes = torch.argmax(cls_output, dim=1)
                    correct = (pred_classes == classification_labels).sum().item()
                    epoch_cls_acc += correct / classification_labels.shape[0]
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                LOGGER.info(f'Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {total_loss.item():.4f}, '
                           f'Cls Acc: {epoch_cls_acc/num_batches:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (imgs, targets, paths, shapes, classification_labels) in enumerate(val_loader):
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device)
                classification_labels = classification_labels.to(device)
                
                # Forward pass
                model_output = model(imgs)
                
                # Compute loss
                total_loss, loss_items = compute_loss(model_output, targets, classification_labels)
                
                # Calculate classification accuracy
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    _, cls_output = model_output
                    if cls_output is not None:
                        pred_classes = torch.argmax(cls_output, dim=1)
                        correct = (pred_classes == classification_labels).sum().item()
                        val_cls_acc += correct / classification_labels.shape[0]
                
                val_loss += total_loss.item()
                val_batches += 1
        
        # Calculate metrics
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_train_cls_acc = epoch_cls_acc / num_batches
        avg_val_cls_acc = val_cls_acc / val_batches
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        LOGGER.info(f'Epoch {epoch}/{epochs} Summary:')
        LOGGER.info(f'  Train Loss: {avg_train_loss:.4f}, Train Cls Acc: {avg_train_cls_acc:.4f}')
        LOGGER.info(f'  Val Loss: {avg_val_loss:.4f}, Val Cls Acc: {avg_val_cls_acc:.4f}')
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        classification_accuracies.append(avg_val_cls_acc)
        
        # Save best model
        fitness_score = avg_val_cls_acc  # Use classification accuracy as fitness
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fitness': fitness_score,
                'hyp': hyp,
            }, f'{project}/{name}/best.pt')
            LOGGER.info(f'New best model saved with fitness: {fitness_score:.4f}')
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fitness': fitness_score,
                'hyp': hyp,
            }, f'{project}/{name}/epoch_{epoch}.pt')
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, classification_accuracies, f'{project}/{name}')
    
    LOGGER.info(f'Training completed. Best fitness: {best_fitness:.4f}')
    return model, best_fitness

def plot_training_curves(train_losses, val_losses, classification_accuracies, save_dir):
    """Plot training curves"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot classification accuracy
    ax2.plot(classification_accuracies, label='Classification Accuracy', color='green')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--project', type=str, default='runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='dual_exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    
    opt = parser.parse_args()
    
    # Create project directory
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=opt.exist_ok)
    
    # Train model
    model, best_fitness = train_dual_model(
        data_yaml=opt.data,
        weights=opt.weights,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        resume=opt.resume,
        hyp_yaml=opt.hyp
    )
    
    print(f'Training completed successfully! Best fitness: {best_fitness:.4f}')
    print(f'Results saved to: {save_dir}')
