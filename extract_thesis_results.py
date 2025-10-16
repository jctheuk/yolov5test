"""
Extract training results from thesis results folders and create an Excel summary.
"""

import os
import pandas as pd
from pathlib import Path
import re

def parse_results_csv(csv_path):
    """Parse results.csv and get the last epoch metrics."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Get the last row (best final epoch)
        last_row = df.iloc[-1]
        
        return {
            'final_epoch': int(last_row['epoch']),
            'train_box_loss': float(last_row['train/box_loss']),
            'train_obj_loss': float(last_row['train/obj_loss']),
            'train_cls_loss': float(last_row['train/cls_loss']),
            'train_cls_task_loss': float(last_row['train/cls_task_loss']),
            'val_box_loss': float(last_row['val/box_loss']),
            'val_obj_loss': float(last_row['val/obj_loss']),
            'val_cls_loss': float(last_row['val/cls_loss']),
            'val_cls_task_loss': float(last_row['val/cls_task_loss']),
            'precision': float(last_row['metrics/precision']),
            'recall': float(last_row['metrics/recall']),
            'mAP_0.5': float(last_row['metrics/mAP_0.5']),
            'mAP_0.5:0.95': float(last_row['metrics/mAP_0.5:0.95']),
        }
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return None

def parse_classification_metrics(txt_path):
    """Parse classification_metrics.txt and get the last epoch metrics."""
    try:
        # Read file and manually parse the header
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (starts with #)
        header_line = None
        for line in lines:
            if line.startswith('#'):
                header_line = line.strip('# \n')
                break
        
        if not header_line:
            return None
        
        # Read the CSV with explicit column names
        df = pd.read_csv(txt_path, comment='#', names=header_line.split(','))
        
        # Get the last row
        last_row = df.iloc[-1]
        
        return {
            'cls_accuracy': float(last_row['accuracy']),
            'cls_precision': float(last_row['precision']),
            'cls_recall': float(last_row['recall']),
            'cls_f1_score': float(last_row['f1_score']),
        }
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
        return None

def extract_model_info(folder_name):
    """Extract model type, architecture, and version from folder name."""
    # Examples: yolov5mlc_p5_v1, yolov5mc_backbone_v2, yolov5sc_p3_v3
    pattern = r'yolov5(mlc|mc|sc)_(backbone|p3|p4|p5)_v(\d+)'
    match = re.match(pattern, folder_name)
    
    if match:
        model_type = match.group(1)
        architecture = match.group(2)
        version = int(match.group(3))
        
        # Model type full name
        model_type_map = {
            'mlc': 'YOLOv5-MLC',
            'mc': 'YOLOv5-MC',
            'sc': 'YOLOv5-SC'
        }
        
        return {
            'model_type': model_type_map.get(model_type, model_type),
            'architecture': architecture,
            'version': f'v{version}',
            'dataset_version': version
        }
    return None

def main():
    base_path = Path('yolov5c/thesis results')
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist!")
        return
    
    results = []
    
    # Get all directories
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    folders = sorted(folders, key=lambda x: x.name)
    
    print(f"Found {len(folders)} folders to process...")
    
    for folder in folders:
        folder_name = folder.name
        print(f"Processing {folder_name}...")
        
        # Extract model info
        model_info = extract_model_info(folder_name)
        if not model_info:
            print(f"  Skipping {folder_name} - couldn't parse folder name")
            continue
        
        # Check for required files
        results_csv = folder / 'results.csv'
        classification_txt = folder / 'classification_metrics.txt'
        
        if not results_csv.exists():
            print(f"  Warning: {results_csv} not found")
            continue
        
        if not classification_txt.exists():
            print(f"  Warning: {classification_txt} not found")
            continue
        
        # Parse results
        detection_metrics = parse_results_csv(results_csv)
        classification_metrics = parse_classification_metrics(classification_txt)
        
        if detection_metrics and classification_metrics:
            result = {
                'folder_name': folder_name,
                **model_info,
                **detection_metrics,
                **classification_metrics
            }
            results.append(result)
            print(f"  [OK] Successfully extracted metrics")
        else:
            print(f"  [FAILED] Failed to extract metrics")
    
    if not results:
        print("\nNo results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'folder_name', 'model_type', 'architecture', 'version', 'dataset_version',
        'final_epoch',
        # Detection metrics
        'precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95',
        # Classification metrics
        'cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1_score',
        # Training losses
        'train_box_loss', 'train_obj_loss', 'train_cls_loss', 'train_cls_task_loss',
        # Validation losses
        'val_box_loss', 'val_obj_loss', 'val_cls_loss', 'val_cls_task_loss',
    ]
    
    df = df[column_order]
    
    # Save to Excel with formatting
    output_file = 'thesis_results_complete.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write main results
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Create summary by model type
        summary_by_model = df.groupby('model_type').agg({
            'mAP_0.5': 'mean',
            'mAP_0.5:0.95': 'mean',
            'cls_accuracy': 'mean',
            'cls_f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        summary_by_model.to_excel(writer, sheet_name='Summary by Model Type')
        
        # Create summary by architecture
        summary_by_arch = df.groupby(['model_type', 'architecture']).agg({
            'mAP_0.5': 'mean',
            'mAP_0.5:0.95': 'mean',
            'cls_accuracy': 'mean',
            'cls_f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        summary_by_arch.to_excel(writer, sheet_name='Summary by Architecture')
        
        # Create summary by dataset version
        summary_by_version = df.groupby(['model_type', 'dataset_version']).agg({
            'mAP_0.5': 'mean',
            'mAP_0.5:0.95': 'mean',
            'cls_accuracy': 'mean',
            'cls_f1_score': 'mean'
        }).round(4)
        summary_by_version.to_excel(writer, sheet_name='Summary by Version')
        
        # Find best models
        best_models = pd.DataFrame({
            'Best mAP@0.5': [df.loc[df['mAP_0.5'].idxmax(), 'folder_name']],
            'Best mAP@0.5:0.95': [df.loc[df['mAP_0.5:0.95'].idxmax(), 'folder_name']],
            'Best Classification Accuracy': [df.loc[df['cls_accuracy'].idxmax(), 'folder_name']],
            'Best Classification F1': [df.loc[df['cls_f1_score'].idxmax(), 'folder_name']],
        }).T
        best_models.columns = ['Model']
        best_models.to_excel(writer, sheet_name='Best Models')
    
    print(f"\n[SUCCESS] Results saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total models processed: {len(results)}")
    print(f"  Model types: {df['model_type'].unique().tolist()}")
    print(f"  Architectures: {df['architecture'].unique().tolist()}")
    print(f"  Dataset versions: {sorted(df['dataset_version'].unique().tolist())}")
    print(f"\nBest Results:")
    print(f"  Best mAP@0.5: {df['mAP_0.5'].max():.4f} ({df.loc[df['mAP_0.5'].idxmax(), 'folder_name']})")
    print(f"  Best mAP@0.5:0.95: {df['mAP_0.5:0.95'].max():.4f} ({df.loc[df['mAP_0.5:0.95'].idxmax(), 'folder_name']})")
    print(f"  Best Classification Accuracy: {df['cls_accuracy'].max():.4f} ({df.loc[df['cls_accuracy'].idxmax(), 'folder_name']})")
    print(f"  Best Classification F1: {df['cls_f1_score'].max():.4f} ({df.loc[df['cls_f1_score'].idxmax(), 'folder_name']})")

if __name__ == '__main__':
    main()

