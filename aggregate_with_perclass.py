"""
Aggregate thesis results WITH per-class metrics where available.

This script:
1. Reads overall metrics from thesis_results_complete.xlsx
2. Extracts available per-class metrics from validation results
3. Generates comprehensive tables with both overall and per-class metrics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_thesis_results(excel_path):
    """Load thesis results from Excel file."""
    df = pd.read_excel(excel_path)
    return df

def aggregate_by_model(df):
    """Aggregate metrics by model type and architecture."""
    # Group by model_type and architecture
    grouped = df.groupby(['model_type', 'architecture']).agg({
        'mAP_0.5': 'mean',
        'mAP_0.5:0.95': 'mean',
        'precision': 'mean',  # Overall detection precision
        'recall': 'mean',      # Overall detection recall
        'cls_accuracy': 'mean',
        'cls_precision': 'mean',
        'cls_recall': 'mean',
        'cls_f1_score': 'mean'
    }).reset_index()
    
    # Convert classification metrics from decimal to percentage (if < 1)
    for col in ['cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1_score']:
        grouped[col] = grouped[col].apply(lambda x: x * 100 if pd.notna(x) and x < 1 else x)
    
    # Convert detection metrics to percentage
    for col in ['precision', 'recall']:
        grouped[col] = grouped[col].apply(lambda x: x * 100 if pd.notna(x) and x < 1 else x)
    
    return grouped

def load_yolov5original_results():
    """Load yolov5original results with per-class data."""
    results = []
    
    models = {
        's': {'avg': 97.78, 'per_class': {'A4C': None, 'PSAX': None}},
        'm': {'avg': 98.25, 'per_class': {'A4C': None, 'PSAX': None}},
        'l': {'avg': 97.37, 'per_class': {'A4C': None, 'PSAX': None}}
    }
    
    # Try to find detailed metrics from val-cls runs
    val_cls_dir = 'yolov5original/runs/val-cls'
    if os.path.exists(val_cls_dir):
        for exp_dir in glob.glob(f"{val_cls_dir}/exp*"):
            detailed_csv = os.path.join(exp_dir, 'detailed_metrics.csv')
            if os.path.exists(detailed_csv):
                try:
                    df = pd.read_csv(detailed_csv)
                    # This has per-class metrics
                    print(f"  Found detailed metrics in {exp_dir}")
                    print(df)
                except:
                    pass
    
    for size, data in models.items():
        results.append({
            'model_type': f'YOLOv5-{size.upper()}',
            'architecture': 'classify',
            'mAP_0.5': None,
            'mAP_0.5:0.95': None,
            'precision': None,
            'recall': None,
            'cls_accuracy': data['avg'],
            'cls_precision': None,
            'cls_recall': None,
            'cls_f1_score': None,
            # Per-class (will be filled if available)
            'A4C_accuracy': data['per_class'].get('A4C'),
            'PSAX_accuracy': data['per_class'].get('PSAX')
        })
    
    return pd.DataFrame(results)

def create_comprehensive_csv(combined, output_path):
    """Create comprehensive CSV with all metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def create_comprehensive_latex(combined, output_path):
    """Create comprehensive LaTeX table."""
    latex_lines = []
    
    # Table header
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Comprehensive YOLOv5 Model Comparison (V1-V5 Averaged)}")
    latex_lines.append(r"\label{tab:comprehensive_comparison}")
    latex_lines.append(r"\scriptsize")  # Use smaller font
    latex_lines.append(r"\begin{tabular}{l|l|cccc|cccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Arch} & \multicolumn{4}{c|}{Detection} & \multicolumn{4}{c}{Classification} \\")
    latex_lines.append(r"\cline{3-10}")
    latex_lines.append(r"& & P & R & mAP@.5 & mAP@.5:.95 & Acc & Prec & Recall & F1 \\")
    latex_lines.append(r"\hline")
    
    # Table rows
    for _, row in combined.iterrows():
        model_name = str(row['model_type']).replace('_', r'\_')
        arch = str(row['architecture']).replace('_', r'\_')
        
        # Detection metrics
        det_p = f"{row['precision']:.1f}" if pd.notna(row.get('precision')) else "--"
        det_r = f"{row['recall']:.1f}" if pd.notna(row.get('recall')) else "--"
        map50 = f"{row['mAP_0.5']:.3f}" if pd.notna(row['mAP_0.5']) else "--"
        map5095 = f"{row['mAP_0.5:0.95']:.3f}" if pd.notna(row['mAP_0.5:0.95']) else "--"
        
        # Classification metrics
        acc = f"{row['cls_accuracy']:.1f}" if pd.notna(row['cls_accuracy']) else "--"
        prec = f"{row['cls_precision']:.1f}" if pd.notna(row['cls_precision']) else "--"
        recall = f"{row['cls_recall']:.1f}" if pd.notna(row['cls_recall']) else "--"
        f1 = f"{row['cls_f1_score']:.1f}" if pd.notna(row['cls_f1_score']) else "--"
        
        # Row
        row_text = f"{model_name} & {arch} & {det_p} & {det_r} & {map50} & {map5095} & {acc} & {prec} & {recall} & {f1} \\\\"
        latex_lines.append(row_text)
    
    # Table footer
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved: {output_path}")

def create_perclass_detection_table(df, output_path):
    """Create per-class detection metrics table (approximation from overall)."""
    # Note: Real per-class detection requires running validation
    # This creates a placeholder structure
    
    rows = []
    for _, row in df.iterrows():
        if pd.notna(row['mAP_0.5']):
            for class_name in ['A4C', 'PSAX']:
                rows.append({
                    'Model': row['model_type'],
                    'Architecture': row['architecture'],
                    'Class': class_name,
                    'Precision': None,  # Need validation to get per-class
                    'Recall': None,
                    'mAP@0.5': None,
                    'mAP@0.5:0.95': None,
                    'Note': 'Run validation for per-class metrics'
                })
    
    if rows:
        perclass_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        perclass_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} (placeholder - run validation for actual data)")

def main():
    base_path = os.getcwd()
    
    print("="*80)
    print("Aggregating metrics with per-class data...")
    print("="*80)
    
    # Load thesis results
    print("\n1. Loading thesis results from Excel...")
    excel_path = os.path.join(base_path, 'thesis_results_complete.xlsx')
    df = load_thesis_results(excel_path)
    print(f"   Loaded {len(df)} records")
    
    # Aggregate
    print("\n2. Aggregating by model and architecture...")
    aggregated = aggregate_by_model(df)
    
    # Load YOLOv5 Original
    print("\n3. Loading YOLOv5 Original results...")
    original_results = load_yolov5original_results()
    
    # Combine
    print("\n4. Combining all results...")
    combined = pd.concat([aggregated, original_results], ignore_index=True)
    combined = combined.sort_values(['model_type', 'architecture'])
    
    # Save comprehensive CSV
    print("\n5. Generating output files...")
    csv_path = 'results/comprehensive_metrics.csv'
    create_comprehensive_csv(combined, csv_path)
    
    # Save comprehensive LaTeX
    tex_path = 'results/comprehensive_table.tex'
    create_comprehensive_latex(combined, tex_path)
    
    # Create per-class detection placeholder
    perclass_det_path = 'results/perclass_detection_placeholder.csv'
    create_perclass_detection_table(combined, perclass_det_path)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    
    print("\nGenerated files:")
    print(f"  [OK] {csv_path}")
    print(f"  [OK] {tex_path}")
    print(f"  [PLACEHOLDER] {perclass_det_path}")
    
    print("\n" + "="*80)
    print("Summary of available metrics:")
    print("="*80)
    
    print(f"\nTotal models: {len(combined)}")
    print("\nOverall Detection Metrics:")
    print(f"  - Precision (overall): Available")
    print(f"  - Recall (overall): Available")
    print(f"  - mAP@0.5: Available")
    print(f"  - mAP@0.5:0.95: Available")
    
    print("\nOverall Classification Metrics:")
    print(f"  - Accuracy (overall): Available")
    print(f"  - Precision (overall): Available")
    print(f"  - Recall (overall): Available")
    print(f"  - F1-Score (overall): Available")
    
    print("\nPer-Class Metrics:")
    print(f"  - Detection (A4C, PSAX): Need to run validation")
    print(f"  - Classification (A4C, PSAX): Need to run validation")
    
    print("\n" + "="*80)
    print("To get per-class metrics:")
    print("="*80)
    print("""
Option 1: Quick check of available per-class data
  python check_available_perclass_data.py

Option 2: Extract per-class metrics (30-60 min)
  python extract_perclass_metrics.py
  
This will run validation on all models and extract:
  • Detection per-class: AP, AR for A4C and PSAX
  • Classification per-class: Acc, Prec, Recall, F1 for A4C and PSAX
""")
    
    print("\n" + "="*80)
    print("Preview of comprehensive results:")
    print("="*80)
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()

