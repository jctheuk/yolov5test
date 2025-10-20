"""
Aggregate YOLOv5 Classification and Detection Results
Creates a comprehensive table with models as rows, classes as subrows
Averages across V1-V5 datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

def extract_classification_metrics(metrics_dir='classification_metrics'):
    """Extract per-class classification metrics from all models"""
    metrics_dir = Path(metrics_dir)
    
    if not metrics_dir.exists():
        print(f"WARNING: Classification metrics directory not found: {metrics_dir}")
        return None
    
    all_data = []
    
    # Find all metrics CSV files
    csv_files = list(metrics_dir.glob('*_metrics.csv'))
    
    for csv_file in csv_files:
        model_name = csv_file.stem.replace('_metrics', '')
        
        # Parse model info
        if 'classifys' in model_name:
            model_size = 'yolov5s'
        elif 'classifym' in model_name:
            model_size = 'yolov5m'
        elif 'classifyl' in model_name:
            model_size = 'yolov5l'
        else:
            continue
        
        version = model_name.split('_')[-1]
        
        # Read metrics
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            all_data.append({
                'Model': model_size,
                'Version': version,
                'Task': 'Classification',
                'Class': row['Class'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1-Score': row['F1-Score'],
                'Support': row['Support']
            })
    
    return pd.DataFrame(all_data) if all_data else None


def extract_detection_metrics_from_results(runs_dir='yolov5c/runs/train'):
    """Extract per-class detection metrics from YOLOv5 results"""
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        print(f"WARNING: Detection runs directory not found: {runs_dir}")
        return None
    
    all_data = []
    
    # Find all experiment directories
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
    
    for exp_dir in exp_dirs:
        results_file = exp_dir / 'results.csv'
        
        if not results_file.exists():
            continue
        
        # Try to parse model info from directory name or opt.yaml
        opt_file = exp_dir / 'opt.yaml'
        model_size = None
        version = None
        
        if opt_file.exists():
            with open(opt_file, 'r') as f:
                import yaml
                opt = yaml.safe_load(f)
                
                # Extract model size from weights or cfg
                if 'weights' in opt:
                    if 'yolov5s' in str(opt['weights']).lower():
                        model_size = 'yolov5s'
                    elif 'yolov5m' in str(opt['weights']).lower():
                        model_size = 'yolov5m'
                    elif 'yolov5l' in str(opt['weights']).lower():
                        model_size = 'yolov5l'
                
                # Extract version from data path
                if 'data' in opt:
                    data_path = str(opt['data'])
                    for v in ['v1', 'v2', 'v3', 'v4', 'v5']:
                        if v.upper() in data_path or v in data_path:
                            version = v
                            break
        
        if not model_size or not version:
            continue
        
        # Read final results (last epoch)
        df = pd.read_csv(results_file)
        last_row = df.iloc[-1]
        
        # Extract per-class metrics if available
        # YOLOv5 results.csv format: epoch, train/*, val/*, metrics/*
        # We need to extract per-class P, R, mAP50, mAP50-95
        
        # For now, extract overall metrics
        metrics = {
            'Model': model_size,
            'Version': version,
            'Task': 'Detection',
            'Class': 'Overall',
            'mAP50': last_row.get('metrics/mAP_0.5', np.nan),
            'mAP50-95': last_row.get('metrics/mAP_0.5:0.95', np.nan),
            'Precision': last_row.get('metrics/precision', np.nan),
            'Recall': last_row.get('metrics/recall', np.nan)
        }
        
        all_data.append(metrics)
    
    return pd.DataFrame(all_data) if all_data else None


def parse_detection_perclass_results(results_dir='yolov5c/runs/train'):
    """Parse per-class detection results from YOLOv5 output files"""
    results_dir = Path(results_dir)
    
    all_data = []
    
    # Look for per-class results in various formats
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for exp_dir in exp_dirs:
        # Check for per-class JSON or text files
        perclass_files = list(exp_dir.glob('*per*class*.json')) + \
                        list(exp_dir.glob('*per*class*.txt'))
        
        for pfile in perclass_files:
            # Parse the file based on format
            # This would need to be customized based on actual file format
            pass
    
    return pd.DataFrame(all_data) if all_data else None


def create_aggregated_table(classification_df, detection_df=None):
    """Create aggregated table with models as rows, classes as subrows"""
    
    if classification_df is None and detection_df is None:
        print("ERROR: No data available")
        return None
    
    results = []
    
    # Process classification data
    if classification_df is not None:
        # Group by model and class, average across versions
        class_grouped = classification_df.groupby(['Model', 'Class']).agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'F1-Score': 'mean',
            'Support': 'mean'
        }).reset_index()
        
        for _, row in class_grouped.iterrows():
            results.append({
                'Model': row['Model'],
                'Task': 'Classification',
                'Class': row['Class'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1-Score': row['F1-Score'],
                'mAP50': np.nan,
                'mAP50-95': np.nan,
                'Accuracy': np.nan,
                'Support': row['Support']
            })
    
    # Process detection data
    if detection_df is not None:
        det_grouped = detection_df.groupby(['Model', 'Class']).agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'mAP50': 'mean',
            'mAP50-95': 'mean'
        }).reset_index()
        
        for _, row in det_grouped.iterrows():
            results.append({
                'Model': row['Model'],
                'Task': 'Detection',
                'Class': row['Class'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1-Score': 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall']) if (row['Precision'] + row['Recall']) > 0 else 0,
                'mAP50': row['mAP50'],
                'mAP50-95': row['mAP50-95'],
                'Accuracy': np.nan,
                'Support': np.nan
            })
    
    df = pd.DataFrame(results)
    
    # Sort by model and task
    df = df.sort_values(['Model', 'Task', 'Class'])
    
    return df


def format_output_table(df, output_dir='classification_metrics'):
    """Format and save output table"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create formatted table
    formatted = df.copy()
    
    # Round numerical columns
    for col in ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-95', 'Accuracy']:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
    
    formatted['Support'] = formatted['Support'].apply(lambda x: f'{int(x)}' if pd.notna(x) else '-')
    
    # Save CSV
    csv_file = output_dir / 'aggregated_results.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"[OK] Saved CSV: {csv_file}")
    
    # Save formatted CSV
    formatted_file = output_dir / 'aggregated_results_formatted.csv'
    formatted.to_csv(formatted_file, index=False)
    print(f"[OK] Saved formatted CSV: {formatted_file}")
    
    # Create LaTeX table
    latex_table = create_latex_table(df)
    latex_file = output_dir / 'aggregated_results.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"[OK] Saved LaTeX: {latex_file}")
    
    # Create Markdown table
    markdown_table = create_markdown_table(formatted)
    md_file = output_dir / 'aggregated_results.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    print(f"[OK] Saved Markdown: {md_file}")
    
    return formatted


def create_latex_table(df):
    """Create LaTeX table with model grouping"""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{YOLOv5 Classification and Detection Results (Averaged V1-V5)}')
    lines.append(r'\label{tab:yolov5_results}')
    lines.append(r'\begin{tabular}{llcccccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & Class & Precision & Recall & F1-Score & mAP50 & mAP50-95 & Support \\')
    lines.append(r'\midrule')
    
    current_model = None
    for _, row in df.iterrows():
        if row['Model'] != current_model:
            if current_model is not None:
                lines.append(r'\midrule')
            current_model = row['Model']
            model_label = row['Model']
        else:
            model_label = ''
        
        # Format class name with task prefix
        class_name = f"{row['Task']}: {row['Class']}"
        
        # Format values
        p = f"{row['Precision']:.4f}" if pd.notna(row['Precision']) else '-'
        r = f"{row['Recall']:.4f}" if pd.notna(row['Recall']) else '-'
        f1 = f"{row['F1-Score']:.4f}" if pd.notna(row['F1-Score']) else '-'
        map50 = f"{row['mAP50']:.4f}" if pd.notna(row['mAP50']) else '-'
        map5095 = f"{row['mAP50-95']:.4f}" if pd.notna(row['mAP50-95']) else '-'
        sup = f"{int(row['Support'])}" if pd.notna(row['Support']) else '-'
        
        lines.append(f"{model_label} & {class_name} & {p} & {r} & {f1} & {map50} & {map5095} & {sup} \\\\")
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)


def create_markdown_table(df):
    """Create Markdown table with model grouping"""
    lines = []
    lines.append("# YOLOv5 Classification and Detection Results")
    lines.append("\n## Aggregated Results (Averaged V1-V5)\n")
    lines.append("| Model | Task | Class | Precision | Recall | F1-Score | mAP50 | mAP50-95 | Support |")
    lines.append("|-------|------|-------|-----------|--------|----------|-------|----------|---------|")
    
    current_model = None
    for _, row in df.iterrows():
        if row['Model'] != current_model:
            current_model = row['Model']
            # Add separator line for new model
            if lines[-1] != "|-------|------|-------|-----------|--------|----------|-------|----------|---------|":
                lines.append("|-------|------|-------|-----------|--------|----------|-------|----------|---------|")
        
        model = row['Model']
        task = row['Task']
        cls = row['Class']
        p = row['Precision']
        r = row['Recall']
        f1 = row['F1-Score']
        map50 = row['mAP50']
        map5095 = row['mAP50-95']
        sup = row['Support']
        
        lines.append(f"| {model} | {task} | {cls} | {p} | {r} | {f1} | {map50} | {map5095} | {sup} |")
    
    return '\n'.join(lines)


def main():
    print("="*80)
    print("YOLOv5 Results Aggregation")
    print("="*80)
    
    # Extract classification metrics
    print("\nExtracting classification metrics...")
    classification_df = extract_classification_metrics()
    
    if classification_df is not None:
        print(f"   Found {len(classification_df)} classification records")
        print(f"   Models: {classification_df['Model'].unique()}")
        print(f"   Versions: {sorted(classification_df['Version'].unique())}")
        print(f"   Classes: {sorted(classification_df['Class'].unique())}")
    else:
        print("   WARNING: No classification data found")
    
    # Extract detection metrics
    print("\nExtracting detection metrics...")
    detection_df = extract_detection_metrics_from_results()
    
    if detection_df is not None:
        print(f"   Found {len(detection_df)} detection records")
    else:
        print("   WARNING: No detection data found")
    
    # Create aggregated table
    print("\nCreating aggregated table...")
    aggregated_df = create_aggregated_table(classification_df, detection_df)
    
    if aggregated_df is None:
        print("ERROR: Failed to create aggregated table")
        return
    
    print(f"   Total rows: {len(aggregated_df)}")
    
    # Format and save output
    print("\nSaving results...")
    formatted_df = format_output_table(aggregated_df)
    
    # Display summary
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    print("\nBy Model:")
    for model in sorted(aggregated_df['Model'].unique()):
        model_data = aggregated_df[aggregated_df['Model'] == model]
        n_classes = len(model_data)
        avg_f1 = model_data['F1-Score'].mean()
        print(f"   {model}: {n_classes} classes, avg F1={avg_f1:.4f}")
    
    print("\nAggregation complete!")
    print("="*80)


if __name__ == '__main__':
    main()

