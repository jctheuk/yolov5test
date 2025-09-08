#!/usr/bin/env python3
"""
Training Diagnostic Script for YOLOv5WithClassification
Collects comprehensive data to diagnose spiking and objectness loss issues
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import yaml
from models.yolo import DetectionModel
from utils.loss import ComputeLoss
from utils.general import check_dataset, check_yaml
from utils.torch_utils import select_device, smart_optimizer
from utils.dataloaders import create_dataloader

class TrainingDiagnostic:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        # Data collection
        self.diagnostic_data = {
            'epochs': [],
            'objectness_data': [],
            'gradient_data': [],
            'loss_breakdown': [],
            'bias_data': [],
            'target_matching': [],
            'classification_data': []
        }
        
    def collect_bias_data(self):
        """Collect bias initialization data"""
        bias_data = {}
        
        # Find Detect layer
        for i, m in enumerate(self.model.model):
            if hasattr(m, 'm') and hasattr(m, 'stride'):  # Detect layer
                for j, mi in enumerate(m.m):
                    if hasattr(mi, 'bias') and mi.bias is not None:
                        b = mi.bias.view(m.na, -1)
                        bias_data[f'layer_{i}_conv_{j}'] = {
                            'objectness_bias': b.data[:, 4].cpu().numpy().tolist(),
                            'classification_bias': b.data[:, 5:5+m.nc].cpu().numpy().tolist(),
                            'stride': m.stride[j].item() if hasattr(m.stride, '__getitem__') else m.stride,
                            'bias_mean': mi.bias.mean().item(),
                            'bias_std': mi.bias.std().item()
                        }
        
        return bias_data
    
    def collect_gradient_data(self):
        """Collect gradient information"""
        grad_data = {
            'grad_norms': [],
            'max_grad_norm': 0.0,
            'gradient_explosion_count': 0,
            'classification_head_grad_norm': 0.0
        }
        
        max_norm = 0.0
        explosion_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_data['grad_norms'].append({
                    'name': name,
                    'norm': grad_norm
                })
                
                max_norm = max(max_norm, grad_norm)
                if grad_norm > 10.0:
                    explosion_count += 1
                    
                # Special attention to classification head
                if 'classification' in name.lower():
                    grad_data['classification_head_grad_norm'] = grad_norm
        
        grad_data['max_grad_norm'] = max_norm
        grad_data['gradient_explosion_count'] = explosion_count
        
        return grad_data
    
    def collect_target_matching_data(self, targets, detection_outputs):
        """Collect target matching information"""
        tcls, tbox, indices, anchors = self.loss_fn.build_targets(detection_outputs, targets)
        
        target_data = {
            'total_targets': sum(len(idx[0]) for idx in indices),
            'targets_per_layer': [len(idx[0]) for idx in indices],
            'target_distribution': {
                'by_layer': {},
                'by_class': {}
            }
        }
        
        # Analyze targets by layer
        for i, idx in enumerate(indices):
            if len(idx[0]) > 0:
                target_data['target_distribution']['by_layer'][f'layer_{i}'] = {
                    'count': len(idx[0]),
                    'class_distribution': torch.bincount(tcls[i]).cpu().numpy().tolist()
                }
        
        return target_data
    
    def collect_loss_breakdown(self, loss_outputs):
        """Collect detailed loss breakdown"""
        if isinstance(loss_outputs, tuple) and len(loss_outputs) >= 4:
            lbox, lobj, lcls, lcls_task = loss_outputs[:4]
            
            loss_data = {
                'box_loss': lbox.item(),
                'objectness_loss': lobj.item(),
                'classification_loss': lcls.item(),
                'classification_task_loss': lcls_task.item(),
                'total_loss': sum([lbox.item(), lobj.item(), lcls.item(), lcls_task.item()])
            }
        else:
            loss_data = {
                'total_loss': loss_outputs.item() if hasattr(loss_outputs, 'item') else float(loss_outputs)
            }
        
        return loss_data
    
    def collect_classification_data(self, classification_output, cls_targets):
        """Collect classification-specific data"""
        if classification_output is None or cls_targets is None:
            return {}
        
        # Convert targets to indices if they're one-hot
        if cls_targets.dim() > 1 and cls_targets.shape[1] > 1:
            cls_targets = cls_targets.argmax(dim=1)
        
        # Calculate predictions
        predictions = torch.softmax(classification_output, dim=1)
        predicted_classes = predictions.argmax(dim=1)
        
        # Calculate accuracy
        accuracy = (predicted_classes == cls_targets).float().mean().item()
        
        classification_data = {
            'output_shape': list(classification_output.shape),
            'output_range': [classification_output.min().item(), classification_output.max().item()],
            'output_mean': classification_output.mean().item(),
            'output_std': classification_output.std().item(),
            'predictions_range': [predictions.min().item(), predictions.max().item()],
            'accuracy': accuracy,
            'class_distribution': {
                'predicted': torch.bincount(predicted_classes).cpu().numpy().tolist(),
                'true': torch.bincount(cls_targets).cpu().numpy().tolist()
            }
        }
        
        return classification_data
    
    def run_diagnostic_epoch(self, dataloader, epoch=0):
        """Run one diagnostic epoch"""
        print(f"\n🔍 Running diagnostic epoch {epoch}")
        
        # Collect bias data (only once)
        if epoch == 0:
            self.diagnostic_data['bias_data'] = self.collect_bias_data()
        
        epoch_data = {
            'epoch': epoch,
            'objectness_data': [],
            'gradient_data': [],
            'loss_breakdown': [],
            'target_matching': [],
            'classification_data': []
        }
        
        self.model.train()
        
        for batch_idx, (imgs, targets, paths, _) in enumerate(dataloader):
            if batch_idx >= 3:  # Only analyze first 3 batches
                break
                
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(imgs)
            
            # Calculate loss
            loss_outputs = self.loss_fn(pred, targets)
            if isinstance(loss_outputs, tuple):
                loss = loss_outputs[0] if len(loss_outputs) > 0 else loss_outputs
            else:
                loss = loss_outputs
            
            # Backward pass
            loss.backward()
            
            # Collect data
            epoch_data['gradient_data'].append(self.collect_gradient_data())
            epoch_data['loss_breakdown'].append(self.collect_loss_breakdown(loss_outputs))
            
            # Extract detection and classification outputs
            if isinstance(pred, tuple) and len(pred) == 2:
                detection_outputs, classification_output = pred
                cls_targets = targets[:, -1] if targets.shape[1] > 5 else None
            else:
                detection_outputs = pred
                classification_output = None
                cls_targets = None
            
            epoch_data['target_matching'].append(
                self.collect_target_matching_data(targets, detection_outputs)
            )
            
            if classification_output is not None:
                epoch_data['classification_data'].append(
                    self.collect_classification_data(classification_output, cls_targets)
                )
            
            # Update optimizer
            self.optimizer.step()
            
            print(f"  Batch {batch_idx}: Loss={loss.item():.6f}")
        
        self.diagnostic_data['epochs'].append(epoch_data)
        return epoch_data
    
    def save_diagnostic_data(self, save_path):
        """Save all diagnostic data"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        with open(save_path / 'diagnostic_data.json', 'w') as f:
            json.dump(self.diagnostic_data, f, indent=2)
        
        # Create summary report
        self.create_summary_report(save_path)
        
        # Create visualizations
        self.create_visualizations(save_path)
        
        print(f"📊 Diagnostic data saved to {save_path}")
    
    def create_summary_report(self, save_path):
        """Create a summary report"""
        report = []
        report.append("# YOLOv5WithClassification Training Diagnostic Report\n")
        
        # Bias analysis
        if 'bias_data' in self.diagnostic_data:
            report.append("## Bias Initialization Analysis")
            for layer_name, bias_info in self.diagnostic_data['bias_data'].items():
                report.append(f"### {layer_name}")
                report.append(f"- Objectness bias mean: {np.mean(bias_info['objectness_bias']):.4f}")
                report.append(f"- Objectness bias std: {np.std(bias_info['objectness_bias']):.4f}")
                report.append(f"- Classification bias mean: {np.mean(bias_info['classification_bias']):.4f}")
                report.append(f"- Stride: {bias_info['stride']}")
                report.append("")
        
        # Gradient analysis
        if self.diagnostic_data['epochs']:
            epoch_data = self.diagnostic_data['epochs'][0]
            if epoch_data['gradient_data']:
                grad_data = epoch_data['gradient_data'][0]
                report.append("## Gradient Analysis")
                report.append(f"- Max gradient norm: {grad_data['max_grad_norm']:.4f}")
                report.append(f"- Gradient explosion count: {grad_data['gradient_explosion_count']}")
                report.append(f"- Classification head grad norm: {grad_data['classification_head_grad_norm']:.4f}")
                report.append("")
        
        # Target matching analysis
        if self.diagnostic_data['epochs']:
            epoch_data = self.diagnostic_data['epochs'][0]
            if epoch_data['target_matching']:
                target_data = epoch_data['target_matching'][0]
                report.append("## Target Matching Analysis")
                report.append(f"- Total targets: {target_data['total_targets']}")
                report.append(f"- Targets per layer: {target_data['targets_per_layer']}")
                report.append("")
        
        # Loss analysis
        if self.diagnostic_data['epochs']:
            epoch_data = self.diagnostic_data['epochs'][0]
            if epoch_data['loss_breakdown']:
                loss_data = epoch_data['loss_breakdown'][0]
                report.append("## Loss Analysis")
                for key, value in loss_data.items():
                    report.append(f"- {key}: {value:.6f}")
                report.append("")
        
        # Save report
        with open(save_path / 'diagnostic_report.md', 'w') as f:
            f.write('\n'.join(report))
    
    def create_visualizations(self, save_path):
        """Create diagnostic visualizations"""
        if not self.diagnostic_data['epochs']:
            return
        
        # Plot loss breakdown
        epochs = [data['epoch'] for data in self.diagnostic_data['epochs']]
        loss_data = [data['loss_breakdown'][0] for data in self.diagnostic_data['epochs']]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Objectness loss
        obj_losses = [loss['objectness_loss'] for loss in loss_data]
        axes[0, 0].plot(epochs, obj_losses, 'b-o')
        axes[0, 0].set_title('Objectness Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Classification task loss
        cls_losses = [loss['classification_task_loss'] for loss in loss_data]
        axes[0, 1].plot(epochs, cls_losses, 'r-o')
        axes[0, 1].set_title('Classification Task Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Gradient norms
        grad_norms = [data['gradient_data'][0]['max_grad_norm'] for data in self.diagnostic_data['epochs']]
        axes[1, 0].plot(epochs, grad_norms, 'g-o')
        axes[1, 0].set_title('Max Gradient Norm')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].grid(True)
        
        # Total targets
        total_targets = [data['target_matching'][0]['total_targets'] for data in self.diagnostic_data['epochs']]
        axes[1, 1].bar(epochs, total_targets, color='orange')
        axes[1, 1].set_title('Total Targets per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Number of Targets')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'diagnostic_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main diagnostic function"""
    # Configuration
    device = select_device('')
    data_yaml = '../Regurgitation-YOLODataset-Detection/data.yaml'
    model_cfg = 'models/yolov5sc.yaml'
    hyp_yaml = 'data/hyps/hyp.fixed_classification_minimal.yaml'
    
    # Load hyperparameters
    with open(hyp_yaml) as f:
        hyp = yaml.safe_load(f)
    
    # Load model
    model = DetectionModel(model_cfg, ch=3, nc=4)
    model = model.to(device)
    
    # Load loss function
    loss_fn = ComputeLoss(model, autobalance=False)
    
    # Load optimizer
    optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], momentum=hyp['momentum'], 
                               decay=hyp['weight_decay'])
    
    # Load dataset
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    train_path = data['train']
    dataloader = create_dataloader(train_path, 416, 32, 32, hyp=hyp, augment=True, 
                                  cache=False, rect=False, rank=-1, workers=8, 
                                  image_weights=False, quad=False, prefix='train')[0]
    
    # Create diagnostic tool
    diagnostic = TrainingDiagnostic(model, loss_fn, optimizer, device)
    
    # Run diagnostics for 3 epochs
    for epoch in range(3):
        diagnostic.run_diagnostic_epoch(dataloader, epoch)
    
    # Save results
    diagnostic.save_diagnostic_data('diagnostic_results')
    
    print("✅ Diagnostic complete!")

if __name__ == '__main__':
    main()
