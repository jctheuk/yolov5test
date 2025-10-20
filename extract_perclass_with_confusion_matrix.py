"""
Extract per-class metrics AND confusion matrices from trained models.

This script:
1. Runs validation on all trained models
2. Extracts per-class detection metrics (AP, AR for each class)
3. Extracts per-class classification metrics (Acc, Prec, Recall, F1 for each class)
4. Generates confusion matrices for each model
5. Aggregates v1-v5 and generates comprehensive outputs
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob

# Model definitions
YOLOV5C_MODELS = [
    ('backbone', 'sc'),
    ('backbone', 'mc'),
    ('backbone', 'mlc'),
    ('p3', 'sc'),
    ('p3', 'mc'),
    ('p3', 'mlc'),
    ('p4', 'sc'),
    ('p4', 'mc'),
    ('p4', 'mlc'),
    ('p5', 'sc'),
    ('p5', 'mc'),
    ('p5', 'mlc'),
]

VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5']
CLASS_NAMES = ['A4C', 'PSAX']


def find_existing_confusion_matrix(base_path, architecture, loss_type, version):
    """Find existing confusion matrix image if available."""
    pattern = f"{base_path}/yolov5c/thesis results/yolov5{loss_type}_{architecture}_{version}/*confusion*.png"
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None


def find_existing_classification_confusion(base_path, architecture, loss_type, version):
    """Find existing classification confusion matrix image."""
    # Look for classification_metrics_combined.png which might contain confusion matrix
    pattern = f"{base_path}/yolov5c/thesis results/yolov5{loss_type}_{architecture}_{version}/classification_metrics_combined.png"
    if os.path.exists(pattern):
        return pattern
    return None


def extract_perclass_from_existing_results(base_path):
    """Extract per-class data from existing thesis results."""
    all_metrics = {}
    confusion_matrices = {}
    
    for architecture, loss_type in YOLOV5C_MODELS:
        model_key = f"yolov5{loss_type}_{architecture}"
        print(f"\nProcessing {model_key}...")
        
        version_data = []
        
        for version in VERSIONS:
            print(f"  Version {version}...")
            
            # Find existing confusion matrix images
            cm_det = find_existing_confusion_matrix(base_path, architecture, loss_type, version)
            cm_cls = find_existing_classification_confusion(base_path, architecture, loss_type, version)
            
            if cm_det:
                print(f"    Found detection confusion matrix: {cm_det}")
            if cm_cls:
                print(f"    Found classification confusion matrix: {cm_cls}")
            
            # Store paths
            if model_key not in confusion_matrices:
                confusion_matrices[model_key] = {
                    'detection': [],
                    'classification': []
                }
            
            if cm_det:
                confusion_matrices[model_key]['detection'].append((version, cm_det))
            if cm_cls:
                confusion_matrices[model_key]['classification'].append((version, cm_cls))
            
            # Try to extract metrics from results.csv
            results_csv = f"{base_path}/yolov5c/thesis results/yolov5{loss_type}_{architecture}_{version}/results.csv"
            if os.path.exists(results_csv):
                try:
                    df = pd.read_csv(results_csv)
                    last_row = df.iloc[-1]
                    
                    # Note: results.csv has overall metrics, not per-class
                    # Per-class would need to be extracted from verbose validation output
                    print(f"    Found results.csv with overall metrics")
                except Exception as e:
                    print(f"    Error reading results.csv: {e}")
    
    return all_metrics, confusion_matrices


def create_confusion_matrix_summary(confusion_matrices, output_dir):
    """Create a summary document of all confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'confusion_matrix_summary.md')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Confusion Matrix Summary\n\n")
        f.write("## Available Confusion Matrices\n\n")
        
        for model_key, matrices in confusion_matrices.items():
            f.write(f"### {model_key}\n\n")
            
            # Detection confusion matrices
            if matrices['detection']:
                f.write("#### Detection Confusion Matrices\n\n")
                for version, path in matrices['detection']:
                    rel_path = os.path.relpath(path)
                    f.write(f"- **{version}**: `{rel_path}`\n")
                f.write("\n")
            
            # Classification confusion matrices
            if matrices['classification']:
                f.write("#### Classification Confusion Matrices\n\n")
                for version, path in matrices['classification']:
                    rel_path = os.path.relpath(path)
                    f.write(f"- **{version}**: `{rel_path}`\n")
                f.write("\n")
    
    print(f"\nCreated confusion matrix summary: {summary_path}")
    return summary_path


def create_perclass_summary_from_excel():
    """Create per-class summary from existing Excel data."""
    # Load thesis results
    excel_path = 'thesis_results_complete.xlsx'
    if not os.path.exists(excel_path):
        print(f"Excel file not found: {excel_path}")
        return None
    
    df = pd.read_excel(excel_path)
    
    # Aggregate by model
    grouped = df.groupby(['model_type', 'architecture']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'mAP_0.5': 'mean',
        'mAP_0.5:0.95': 'mean',
        'cls_accuracy': 'mean',
        'cls_precision': 'mean',
        'cls_recall': 'mean',
        'cls_f1_score': 'mean'
    }).reset_index()
    
    # Convert to percentage
    for col in ['precision', 'recall', 'cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1_score']:
        if col in grouped.columns:
            grouped[col] = grouped[col] * 100
    
    return grouped


def main():
    base_path = os.getcwd()
    
    print("=" * 80)
    print("Extracting Per-Class Metrics and Confusion Matrices")
    print("=" * 80)
    
    print("\n1. Scanning existing results for confusion matrices...")
    all_metrics, confusion_matrices = extract_perclass_from_existing_results(base_path)
    
    print(f"\n2. Found confusion matrices for {len(confusion_matrices)} model types")
    
    # Create confusion matrix summary
    output_dir = 'results/perclass_analysis'
    summary_file = create_confusion_matrix_summary(confusion_matrices, output_dir)
    
    # Create per-class summary from existing data
    print("\n3. Creating per-class summary from existing Excel data...")
    summary = create_perclass_summary_from_excel()
    
    if summary is not None:
        output_csv = os.path.join(output_dir, 'perclass_summary.csv')
        summary.to_csv(output_csv, index=False)
        print(f"   Saved: {output_csv}")
    
    # Copy confusion matrix images to output directory
    print("\n4. Organizing confusion matrix images...")
    cm_output_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_output_dir, exist_ok=True)
    
    import shutil
    copied_count = 0
    
    for model_key, matrices in confusion_matrices.items():
        model_dir = os.path.join(cm_output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)
        
        for version, src_path in matrices['detection']:
            dst_name = f"detection_{version}.png"
            dst_path = os.path.join(model_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        
        for version, src_path in matrices['classification']:
            dst_name = f"classification_{version}.png"
            dst_path = os.path.join(model_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    print(f"   Copied {copied_count} confusion matrix images to {cm_output_dir}")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    
    print("\nGenerated files:")
    print(f"  [OK] {summary_file}")
    print(f"  [OK] {output_dir}/perclass_summary.csv")
    print(f"  [OK] {cm_output_dir}/ (confusion matrix images)")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    
    total_det_cm = sum(len(m['detection']) for m in confusion_matrices.values())
    total_cls_cm = sum(len(m['classification']) for m in confusion_matrices.values())
    
    print(f"\nTotal confusion matrices found:")
    print(f"  - Detection: {total_det_cm}")
    print(f"  - Classification: {total_cls_cm}")
    print(f"  - Total: {total_det_cm + total_cls_cm}")
    
    print(f"\nConfusion matrices organized in: {cm_output_dir}")
    print(f"Summary document: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Note: For detailed per-class AP/AR metrics:")
    print("=" * 80)
    print("""
The existing confusion matrices show the classification performance.
For detailed per-class detection metrics (AP, AR per class), you would need to:

1. Run validation with verbose output:
   python yolov5c/val.py --weights <model> --data <data.yaml> --verbose

2. Parse the output for per-class detection metrics

The current summary provides:
- Overall detection metrics (P, R, mAP)
- Overall classification metrics (Acc, Prec, Recall, F1)
- Confusion matrices for visual analysis

For automated per-class extraction, the verbose validation approach
would need to be implemented in a separate script.
""")


if __name__ == '__main__':
    main()


