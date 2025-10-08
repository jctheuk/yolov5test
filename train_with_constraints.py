#!/usr/bin/env python3
"""
Training script with anatomical constraints
Demonstrates how to use hidden rules to improve mAP and accuracy
"""

import os
import sys
import torch
import argparse
import yaml
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def modify_hyperparameters(hyp_path, enable_constraints, constraint_weight, constraint_mode):
    """Modify hyperparameter file with constraint settings"""
    try:
        # Read current hyperparameters
        with open(hyp_path, 'r') as f:
            hyp_data = yaml.safe_load(f)
        
        # Update constraint settings
        hyp_data['use_anatomical_constraints'] = enable_constraints
        if constraint_weight is not None:
            hyp_data['constraint_weight'] = constraint_weight
        hyp_data['constraint_mode'] = constraint_mode
        
        # Write back to file
        with open(hyp_path, 'w') as f:
            yaml.dump(hyp_data, f, default_flow_style=False)
        
        print(f"✅ Updated hyperparameters: constraints={enable_constraints}, weight={constraint_weight}, mode={constraint_mode}")
        
    except Exception as e:
        print(f"❌ Failed to modify hyperparameters: {e}")

def main():
    """Main training function with anatomical constraints"""
    
    # Training arguments
    parser = argparse.ArgumentParser(description='Train YOLOv5 with Anatomical Constraints')
    parser.add_argument('--data', type=str, default='regurgitationV1/data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='yolov5c/data/hyps/hyp.constraint_priority.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--project', default='runs/constraint_training', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    # Anatomical constraints arguments
    parser.add_argument('--enable-constraints', action='store_true', default=True, help='enable anatomical constraints (default: True)')
    parser.add_argument('--disable-constraints', action='store_true', help='disable anatomical constraints')
    parser.add_argument('--constraint-weight', type=float, default=None, help='weight for constraint loss (overrides hyperparameter file)')
    parser.add_argument('--constraint-mode', type=str, default='soft', choices=['soft', 'strict', 'mixed'], help='constraint enforcement mode: soft, strict, or mixed')
    parser.add_argument('--show-constraints', action='store_true', help='show anatomical constraints and exit')
    
    args = parser.parse_args()
    
    # Handle constraint-specific arguments
    if args.disable_constraints:
        args.enable_constraints = False
    
    # Show constraints and exit if requested
    if args.show_constraints:
        show_constraints_info()
        return
    
    print("=" * 60)
    print("ANATOMICAL CONSTRAINTS TRAINING")
    print("=" * 60)
    print("Leveraging hidden rules to improve mAP and accuracy")
    print()
    
    # Display constraint configuration
    print("Constraint Configuration:")
    print(f"  Enable Constraints: {args.enable_constraints}")
    print(f"  Constraint Mode: {args.constraint_mode}")
    if args.constraint_weight is not None:
        print(f"  Constraint Weight: {args.constraint_weight} (override)")
    else:
        print(f"  Constraint Weight: (from hyperparameter file)")
    print()
    
    # Check if dataset exists
    if not os.path.exists(args.data):
        print(f"Error: Dataset file {args.data} not found!")
        return
    
    # Check if hyperparameter file exists
    if not os.path.exists(args.hyp):
        print(f"Error: Hyperparameter file {args.hyp} not found!")
        return
    
    # Import training functions
    try:
        from yolov5c.train import main as train_main, parse_opt
        print("✅ Successfully imported YOLOv5c training function")
    except ImportError as e:
        print(f"❌ Failed to import training function: {e}")
        return
    
    # Training configuration
    print(f"Training Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Hyperparameters: {args.hyp}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Project: {args.project}")
    print(f"  Name: {args.name}")
    print()
    
    # Check anatomical constraints
    print("Checking Anatomical Constraints:")
    print("-" * 40)
    try:
        from yolov5c.utils.anatomical_constraints import AnatomicalConstraints
        constraints = AnatomicalConstraints()
        constraints.print_constraints()
        print("✅ Anatomical constraints loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load anatomical constraints: {e}")
        return
    
    print()
    print("Starting Training with Anatomical Constraints...")
    print("=" * 60)
    
    # Start training
    try:
        # Create training arguments
        sys.argv = [
            'train_with_constraints.py',
            '--data', args.data,
            '--hyp', args.hyp,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--weights', args.weights,
            '--project', args.project,
            '--name', args.name
        ]
        
        # Modify hyperparameter file with constraint settings
        if args.constraint_weight is not None or not args.enable_constraints:
            modify_hyperparameters(args.hyp, args.enable_constraints, args.constraint_weight, args.constraint_mode)
        
        if args.exist_ok:
            sys.argv.append('--exist-ok')
        if args.resume:
            sys.argv.append('--resume')
        
        # Parse arguments and run training
        opt = parse_opt()
        train_main(opt)
        
        print()
        print("=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print("Check the results in the runs/constraint_training directory")
        print("The model should show improved mAP due to anatomical constraints")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def show_constraints_info():
    """Show anatomical constraints information"""
    print("=" * 60)
    print("ANATOMICAL CONSTRAINTS INFORMATION")
    print("=" * 60)
    
    try:
        from yolov5c.utils.anatomical_constraints import AnatomicalConstraints
        constraints = AnatomicalConstraints()
        constraints.print_constraints()
        
        print("\nCONSTRAINT MODES:")
        print("-" * 40)
        print("• soft: Apply weighted penalties for violations (default)")
        print("• strict: Completely forbid impossible detections")
        print("• mixed: Combine soft and strict constraints")
        
        print("\nUSAGE EXAMPLES:")
        print("-" * 40)
        print("• Enable constraints: --enable-constraints")
        print("• Disable constraints: --disable-constraints")
        print("• Set constraint weight: --constraint-weight 0.5")
        print("• Set constraint mode: --constraint-mode strict")
        print("• Show this info: --show-constraints")
        
    except Exception as e:
        print(f"❌ Failed to load constraints: {e}")

def show_usage_examples():
    """Show usage examples for constraint training"""
    print("USAGE EXAMPLES:")
    print("=" * 50)
    print()
    print("1. Basic training with constraints:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml")
    print()
    print("2. Training with custom hyperparameters:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --hyp yolov5c/data/hyps/hyp.constraint_priority.yaml")
    print()
    print("3. Training with specific epochs and batch size:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --epochs 100 --batch-size 8")
    print()
    print("4. Training with specific device:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --device 0")
    print()
    print("5. Resume training:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --resume")
    print()
    print("6. Constraint-specific options:")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --constraint-weight 0.5 --constraint-mode strict")
    print("   python train_with_constraints.py --data regurgitationV1/data.yaml --disable-constraints")
    print("   python train_with_constraints.py --show-constraints")
    print()
    print("EXPECTED IMPROVEMENTS:")
    print("=" * 50)
    print("- Higher mAP due to anatomically consistent predictions")
    print("- Better classification accuracy for view types")
    print("- Reduced false positives from impossible detections")
    print("- More stable training with constraint regularization")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_usage_examples()
    else:
        main()
