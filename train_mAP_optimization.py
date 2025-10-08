#!/usr/bin/env python3
"""
mAP Optimization Training Script
Target: 80% mAP@0.5 for echocardiogram regurgitation detection
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_training_experiment(config_name, epochs, batch_size, device, hyp_file):
    """Run a training experiment with specific configuration"""
    
    print(f"\n{'='*60}")
    print(f"Starting mAP Optimization Experiment: {config_name}")
    print(f"{'='*60}")
    
    # Clear dataset caches before training
    print("Clearing dataset caches...")
    clear_dataset_caches()
    
    # Training command
    cmd = [
        "python", "yolov5c/train.py",
        "--data", "regurgitationV1/data.yaml",
        "--hyp", hyp_file,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device,
        "--name", f"mAP_optimization_{config_name}",
        "--patience", "0",  # Disable early stopping
        "--cache", "ram",  # Use RAM caching for speed
        "--workers", "8",  # Optimize data loading
        "--optimizer", "AdamW",  # Use AdamW for better convergence
        "--cos-lr",  # Cosine learning rate schedule
        "--save-period", "50",  # Save every 50 epochs
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting training...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def clear_dataset_caches():
    """Clear dataset caches for fresh training"""
    dataset = "regurgitationV1"
    sets = ["train", "valid", "test"]
    
    for d in sets:
        labels_path = Path(dataset) / d / "labels"
        if labels_path.exists():
            cache_files = list(labels_path.glob("*.cache*"))
            for cache_file in cache_files:
                cache_file.unlink()
                print(f"   Removed: {cache_file}")

def main():
    parser = argparse.ArgumentParser(description='mAP Optimization Training')
    parser.add_argument('--config', type=str, default='all', 
                       choices=['all', 'optimized', 'augmentation', 'baseline'],
                       help='Which configuration to run')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Training configurations
    configs = {
        'optimized': {
            'hyp_file': 'yolov5c/data/hyps/hyp.mAP_optimized.yaml',
            'description': 'Aggressive optimization with higher learning rates and loss weights'
        },
        'augmentation': {
            'hyp_file': 'yolov5c/data/hyps/hyp.medical_augmentation.yaml', 
            'description': 'Medical image augmentation strategy'
        },
        'baseline': {
            'hyp_file': 'yolov5c/data/hyps/hyp.constraint_priority.yaml',
            'description': 'Baseline constraint training for comparison'
        }
    }
    
    if args.config == 'all':
        # Run all configurations
        for config_name, config_info in configs.items():
            print(f"\nConfiguration: {config_name}")
            print(f"Description: {config_info['description']}")
            
            success = run_training_experiment(
                config_name=config_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                hyp_file=config_info['hyp_file']
            )
            
            if not success:
                print(f"Failed to complete {config_name} experiment")
                break
                
    else:
        # Run single configuration
        if args.config in configs:
            config_info = configs[args.config]
            print(f"🎯 Configuration: {args.config}")
            print(f"📝 Description: {config_info['description']}")
            
            run_training_experiment(
                config_name=args.config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                hyp_file=config_info['hyp_file']
            )
        else:
            print(f"Unknown configuration: {args.config}")
            return
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Check results in yolov5c/runs/train/mAP_optimization_*")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
