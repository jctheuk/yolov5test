"""
Extract per-class detection metrics from YOLOv5 thesis results
"""

import pandas as pd
import json
from pathlib import Path
import re

def parse_classification_metrics(txt_file):
    """Parse classification metrics from txt file"""
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    # Get last line (final epoch)
    if len(lines) > 1:
        last_line = lines[-1].strip()
        parts = last_line.split(',')
        if len(parts) >= 4:
            return {
                'accuracy': float(parts[1]),
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1_score': float(parts[4]) if len(parts) > 4 else 0
            }
    return None

def parse_results_csv(csv_file):
    """Parse detection results from results.csv"""
    df = pd.read_csv(csv_file)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Get last epoch
    last_row = df.iloc[-1]
    
    # Use bracket notation for column names with special characters
    return {
        'precision': last_row['metrics/precision'] if 'metrics/precision' in df.columns else 0,
        'recall': last_row['metrics/recall'] if 'metrics/recall' in df.columns else 0,
        'mAP50': last_row['metrics/mAP_0.5'] if 'metrics/mAP_0.5' in df.columns else 0,
        'mAP50-95': last_row['metrics/mAP_0.5:0.95'] if 'metrics/mAP_0.5:0.95' in df.columns else 0,
    }

def extract_all_thesis_results(base_dir='yolov5c/thesis results'):
    """Extract all metrics from thesis results"""
    base_dir = Path(base_dir)
    
    all_results = []
    
    # Find all result directories
    result_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for result_dir in result_dirs:
        dir_name = result_dir.name
        
        # Parse directory name: {model}_{version}
        # Examples: yolov5sc_backbone_v1, yolov5mc_p3_v2
        parts = dir_name.split('_')
        
        if len(parts) < 2:
            continue
        
        # Extract model type
        model_prefix = parts[0]  # yolov5sc, yolov5mc, yolov5mlc
        
        # Map to standard names
        if 'sc' in model_prefix:
            model_size = 'yolov5s'
            loss_type = 'SC'
        elif 'mlc' in model_prefix:
            model_size = 'yolov5m'
            loss_type = 'MLC'
        elif 'mc' in model_prefix:
            model_size = 'yolov5m'
            loss_type = 'MC'
        else:
            continue
        
        # Extract architecture (backbone, p3, p4, p5)
        arch = None
        for part in parts:
            if part in ['backbone', 'p3', 'p4', 'p5']:
                arch = part
                break
        
        # Extract version
        version = None
        for part in parts:
            if part.startswith('v') and len(part) == 2:
                version = part
                break
        
        if not arch or not version:
            continue
        
        # Parse metrics
        class_metrics_file = result_dir / 'classification_metrics.txt'
        results_file = result_dir / 'results.csv'
        
        if not results_file.exists():
            continue
        
        det_metrics = parse_results_csv(results_file)
        
        class_metrics = None
        if class_metrics_file.exists():
            class_metrics = parse_classification_metrics(class_metrics_file)
        
        result = {
            'model_size': model_size,
            'loss_type': loss_type,
            'architecture': arch,
            'version': version,
            'det_precision': det_metrics['precision'],
            'det_recall': det_metrics['recall'],
            'det_mAP50': det_metrics['mAP50'],
            'det_mAP50-95': det_metrics['mAP50-95'],
        }
        
        if class_metrics:
            result.update({
                'cls_accuracy': class_metrics['accuracy'],
                'cls_precision': class_metrics['precision'],
                'cls_recall': class_metrics['recall'],
                'cls_f1_score': class_metrics['f1_score']
            })
        
        all_results.append(result)
    
    return pd.DataFrame(all_results)

def aggregate_by_model_class(df):
    """Aggregate results by model and architecture, average across versions"""
    
    # Group by model, loss, and architecture
    grouped = df.groupby(['model_size', 'loss_type', 'architecture']).agg({
        'det_precision': 'mean',
        'det_recall': 'mean',
        'det_mAP50': 'mean',
        'det_mAP50-95': 'mean',
        'cls_accuracy': 'mean',
        'cls_precision': 'mean',
        'cls_recall': 'mean',
        'cls_f1_score': 'mean'
    }).reset_index()
    
    return grouped

def main():
    print("="*80)
    print("Extracting Detection Per-Class Metrics from Thesis Results")
    print("="*80)
    
    # Extract all results
    print("\nExtracting metrics from thesis results...")
    df = extract_all_thesis_results()
    
    if df.empty:
        print("ERROR: No results found")
        return
    
    print(f"   Found {len(df)} experiment results")
    print(f"   Models: {df['model_size'].unique()}")
    print(f"   Loss types: {df['loss_type'].unique()}")
    print(f"   Architectures: {df['architecture'].unique()}")
    print(f"   Versions: {sorted(df['version'].unique())}")
    
    # Aggregate by model and architecture
    print("\nAggregating across versions...")
    aggregated = aggregate_by_model_class(df)
    
    print(f"   Aggregated to {len(aggregated)} configurations")
    
    # Save results
    output_dir = Path('classification_metrics')
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    df.to_csv(output_dir / 'detection_results_detailed.csv', index=False, float_format='%.6f')
    print(f"\n[OK] Saved: {output_dir / 'detection_results_detailed.csv'}")
    
    # Save aggregated results
    aggregated.to_csv(output_dir / 'detection_results_aggregated.csv', index=False, float_format='%.6f')
    print(f"[OK] Saved: {output_dir / 'detection_results_aggregated.csv'}")
    
    # Display summary
    print("\n" + "="*80)
    print("Summary by Model and Architecture")
    print("="*80)
    
    for model in sorted(aggregated['model_size'].unique()):
        model_data = aggregated[aggregated['model_size'] == model]
        print(f"\n{model.upper()}:")
        for _, row in model_data.iterrows():
            print(f"  {row['loss_type']}-{row['architecture']:>8}: "
                  f"mAP50={row['det_mAP50']:.4f}, "
                  f"cls_acc={row['cls_accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("Extraction complete!")
    print("="*80)

if __name__ == '__main__':
    main()

