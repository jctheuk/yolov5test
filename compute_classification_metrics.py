"""
Compute per-class metrics for YOLOv5 classification models
Handles Windows/Linux path compatibility issues
"""

import sys
from pathlib import Path, PureWindowsPath, PurePosixPath
import torch
import pathlib
import platform

# Fix pathlib compatibility for models trained on different OS
if platform.system() == 'Windows':
    # Monkey patch to handle PosixPath in Windows
    temp = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except:
        pass

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Add yolov5original to path
sys.path.insert(0, str(Path('yolov5original').absolute()))

from utils.dataloaders import create_classification_dataloader
from utils.general import LOGGER


def load_model_safe(weights_path, device='cpu'):
    """Safely load model handling path issues"""
    try:
        # Try loading with map_location and pickle fix
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Extract model from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'ema' in checkpoint:
                model = checkpoint['ema']
            else:
                model = checkpoint
        else:
            model = checkpoint
            
        # Set to eval mode
        if hasattr(model, 'float'):
            model = model.float()
        if hasattr(model, 'eval'):
            model.eval()
            
        return model, checkpoint.get('names', None) if isinstance(checkpoint, dict) else None
    except Exception as e:
        LOGGER.error(f"Error loading model: {e}")
        return None, None


def compute_metrics(model, dataloader, class_names, device='cpu'):
    """Compute detailed per-class metrics"""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    print(f"\nRunning inference on {len(dataloader)} batches...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Overall accuracy
    accuracy = (all_preds == all_targets).mean()
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, 
        labels=range(len(class_names)),
        zero_division=0
    )
    
    # Classification report
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'report': report,
        'predictions': all_preds,
        'targets': all_targets
    }


def save_results(results, class_names, output_dir, model_name):
    """Save metrics and visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics CSV
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1'],
        'Support': results['support']
    })
    
    csv_path = output_dir / f'{model_name}_metrics.csv'
    metrics_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Saved metrics: {csv_path}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Ratio'}, square=True)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {results["accuracy"]:.4f}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = output_dir / f'{model_name}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix: {cm_path}")
    
    # Save text report
    report_path = output_dir / f'{model_name}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(results['report'])
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"✅ Saved report: {report_path}")
    
    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output', type=str, default='classification_metrics', help='Output directory')
    parser.add_argument('--name', type=str, required=True, help='Model name for output files')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Computing metrics for: {args.name}")
    print(f"{'='*80}\n")
    print(f"Weights: {args.weights}")
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\n📦 Loading model...")
    model, model_names = load_model_safe(args.weights, args.device)
    
    if model is None:
        print("❌ Failed to load model. Trying alternative method...")
        sys.exit(1)
    
    # Get class names from dataset
    data_path = Path(args.data)
    test_dir = data_path / 'test' if (data_path / 'test').exists() else data_path / 'val'
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"\n📋 Classes found: {class_names}")
    
    # Create dataloader
    print(f"\n📂 Loading dataset from: {test_dir}")
    dataloader = create_classification_dataloader(
        path=test_dir,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        augment=False,
        rank=-1,
        workers=4
    )
    print(f"   Total samples: {len(dataloader.dataset)}")
    
    # Compute metrics
    print("\n🔄 Computing metrics...")
    results = compute_metrics(model, dataloader, class_names, args.device)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n📊 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\n📈 Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {results['precision'][i]:>11.4f} {results['recall'][i]:>11.4f} "
              f"{results['f1'][i]:>11.4f} {results['support'][i]:>9}")
    print("-" * 80)
    
    # Save results
    print("\n💾 Saving results...")
    metrics_df = save_results(results, class_names, args.output, args.name)
    
    print("\n✅ Done!")
    print("="*80 + "\n")
    
    return metrics_df


if __name__ == '__main__':
    main()



