"""
Create final comprehensive table with architecture-based grouping
Rows: Model+Architecture (e.g., yolov5s_p5)
Subrows: Classification classes (A4C, PLAX, PSAX) + Detection classes (AR, MR, TR, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_classification_data(metrics_dir='classification_metrics'):
    """Load all classification metrics (already averaged across V1-V5)"""
    metrics_dir = Path(metrics_dir)
    
    all_data = []
    csv_files = list(metrics_dir.glob('classify*_v*_metrics.csv'))
    
    for csv_file in csv_files:
        model_name = csv_file.stem.replace('_metrics', '')
        
        # Parse model info
        if 'classifys' in model_name:
            model_size = 's'
        elif 'classifym' in model_name:
            model_size = 'm'
        elif 'classifyl' in model_name:
            model_size = 'l'
        else:
            continue
        
        version = model_name.split('_')[-1]
        
        # Read metrics
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            all_data.append({
                'model_size': model_size,
                'version': version,
                'class_name': row['Class'],
                'task_type': 'Classification',
                'precision': row['Precision'],
                'recall': row['Recall'],
                'f1_score': row['F1-Score'],
                'support': row['Support']
            })
    
    return pd.DataFrame(all_data)

def load_detection_data(detection_file='classification_metrics/detection_results_aggregated.csv'):
    """Load aggregated detection data"""
    if not Path(detection_file).exists():
        print(f"WARNING: Detection file not found: {detection_file}")
        return None
    
    df = pd.read_csv(detection_file)
    
    # Rename columns to match classification format
    df = df.rename(columns={
        'model_size': 'model_size',
        'loss_type': 'loss_type',
        'architecture': 'architecture',
        'det_precision': 'precision',
        'det_recall': 'recall',
        'det_mAP50': 'mAP50',
        'det_mAP50-95': 'mAP50_95',
        'cls_accuracy': 'cls_accuracy',
        'cls_precision': 'cls_precision',
        'cls_recall': 'cls_recall',
        'cls_f1_score': 'cls_f1'
    })
    
    # Extract model size letter (yolov5s -> s)
    df['model_size'] = df['model_size'].str.replace('yolov5', '')
    
    df['task_type'] = 'Detection'
    df['class_name'] = 'Overall'  # Detection doesn't have per-class breakdown
    df['f1_score'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
    
    return df

def create_comprehensive_table(class_df, det_df):
    """Create final table with model+arch as main rows"""
    
    # For classification: average across V1-V5
    class_avg = class_df.groupby(['model_size', 'class_name']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'support': 'mean'
    }).reset_index()
    
    class_avg['architecture'] = 'original'  # Original YOLOv5 classification
    class_avg['task_type'] = 'Classification'
    class_avg['mAP50'] = np.nan
    class_avg['mAP50_95'] = np.nan
    class_avg['cls_accuracy'] = np.nan
    
    # For detection: already averaged
    det_formatted = det_df[[
        'model_size', 'architecture', 'class_name', 'task_type',
        'precision', 'recall', 'f1_score', 'mAP50', 'mAP50_95', 'cls_accuracy'
    ]].copy()
    det_formatted['support'] = np.nan
    
    # Combine
    combined = pd.concat([class_avg, det_formatted], ignore_index=True)
    
    # Sort by model_size, architecture, task_type, class_name
    combined = combined.sort_values(['model_size', 'architecture', 'task_type', 'class_name'])
    
    # Create model_config column (e.g., yolov5s_p5)
    combined['model_config'] = 'yolov5' + combined['model_size'] + '_' + combined['architecture']
    
    return combined

def format_markdown_table(df, output_file='docs/FINAL_COMPREHENSIVE_TABLE.md'):
    """Format as markdown table"""
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    
    lines = []
    lines.append("# YOLOv5 Classification and Detection Comprehensive Results")
    lines.append("")
    lines.append("## Aggregated Performance Across V1-V5 Datasets")
    lines.append("")
    lines.append("### Table Structure")
    lines.append("- **Main Rows**: Model configurations (yolov5s_p5, yolov5m_backbone, etc.)")
    lines.append("- **Sub-rows**: Classification classes (A4C, PLAX, PSAX) + Detection metrics (Overall)")
    lines.append("- **Columns**: Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95, Classification Accuracy")
    lines.append("")
    lines.append("### Complete Results Table")
    lines.append("")
    lines.append("| 模型配置 | 類別/任務 | 精確率 | 召回率 | F1分數 | mAP@0.5 | mAP@0.5:0.95 | 分類準確率 | 樣本數 |")
    lines.append("|---------|---------|--------|--------|--------|---------|------------|-----------|--------|")
    
    current_config = None
    for _, row in df.iterrows():
        config = row['model_config']
        task = row['task_type']
        class_name = row['class_name']
        
        # Format values
        p = f"{row['precision']:.4f}" if pd.notna(row['precision']) else "-"
        r = f"{row['recall']:.4f}" if pd.notna(row['recall']) else "-"
        f1 = f"{row['f1_score']:.4f}" if pd.notna(row['f1_score']) else "-"
        map50 = f"{row['mAP50']:.4f}" if pd.notna(row['mAP50']) else "-"
        map5095 = f"{row['mAP50_95']:.4f}" if pd.notna(row['mAP50_95']) else "-"
        cls_acc = f"{row['cls_accuracy']:.4f}" if pd.notna(row['cls_accuracy']) else "-"
        support = f"{int(row['support'])}" if pd.notna(row['support']) else "-"
        
        # Add separator for new configuration
        if config != current_config:
            if current_config is not None:
                lines.append("|---------|---------|--------|--------|--------|---------|------------|-----------|--------|")
            current_config = config
        
        # Format class/task name
        if task == 'Classification':
            task_label = f"{class_name} (分類)"
        else:
            task_label = f"{class_name} (檢測)"
        
        lines.append(f"| {config} | {task_label} | {p} | {r} | {f1} | {map50} | {map5095} | {cls_acc} | {support} |")
    
    # Add summary statistics
    lines.append("")
    lines.append("## 模型配置說明")
    lines.append("")
    lines.append("### 分類模型（Classification）")
    lines.append("- **yolov5s_original**: YOLOv5-Small 原始分類模型")
    lines.append("- **yolov5m_original**: YOLOv5-Medium 原始分類模型")
    lines.append("- **yolov5l_original**: YOLOv5-Large 原始分類模型")
    lines.append("")
    lines.append("### 檢測模型（Detection）")
    lines.append("- **yolov5s_SC**: Small模型 + Simple Classification Loss")
    lines.append("  - backbone, p3, p4, p5: 不同檢測頭配置")
    lines.append("- **yolov5m_MC**: Medium模型 + Multi-scale Classification Loss")
    lines.append("  - backbone, p3, p4, p5: 不同檢測頭配置")
    lines.append("- **yolov5m_MLC**: Medium模型 + Modified Loss with Constraints")
    lines.append("  - backbone, p3, p4, p5: 不同檢測頭配置")
    lines.append("")
    lines.append("## 性能總結")
    lines.append("")
    
    # Add performance summary
    class_models = df[df['task_type'] == 'Classification'].groupby('model_size')['f1_score'].mean()
    det_models = df[df['task_type'] == 'Detection'].groupby('model_size')['mAP50'].mean()
    
    lines.append("### 分類任務表現")
    lines.append("")
    lines.append("| 模型大小 | 平均F1分數 |")
    lines.append("|---------|-----------|")
    for model in sorted(class_models.index):
        lines.append(f"| yolov5{model} | {class_models[model]:.4f} |")
    
    lines.append("")
    lines.append("### 檢測任務表現")
    lines.append("")
    lines.append("| 模型大小 | 平均mAP@0.5 |")
    lines.append("|---------|------------|")
    for model in sorted(det_models.index):
        lines.append(f"| yolov5{model} | {det_models[model]:.4f} |")
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"[OK] Saved comprehensive table: {output_file}")
    return output_file

def main():
    print("="*80)
    print("Creating Final Comprehensive Table")
    print("="*80)
    
    # Load classification data
    print("\nLoading classification metrics...")
    class_df = load_classification_data()
    print(f"   Loaded {len(class_df)} classification records")
    print(f"   Models: {sorted(class_df['model_size'].unique())}")
    print(f"   Versions: {sorted(class_df['version'].unique())}")
    print(f"   Classes: {sorted(class_df['class_name'].unique())}")
    
    # Load detection data
    print("\nLoading detection metrics...")
    det_df = load_detection_data()
    if det_df is not None:
        print(f"   Loaded {len(det_df)} detection configurations")
        print(f"   Models: {sorted(det_df['model_size'].unique())}")
        print(f"   Architectures: {sorted(det_df['architecture'].unique())}")
    
    # Create comprehensive table
    print("\nCreating comprehensive table...")
    combined = create_comprehensive_table(class_df, det_df)
    print(f"   Total rows: {len(combined)}")
    
    # Save as markdown
    print("\nFormatting and saving...")
    md_file = format_markdown_table(combined)
    
    # Also save as CSV
    csv_file = Path('docs/FINAL_COMPREHENSIVE_TABLE.csv')
    csv_file.parent.mkdir(exist_ok=True)
    combined.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"[OK] Saved CSV: {csv_file}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nTotal configurations: {combined['model_config'].nunique()}")
    print(f"Classification entries: {len(combined[combined['task_type'] == 'Classification'])}")
    print(f"Detection entries: {len(combined[combined['task_type'] == 'Detection'])}")
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

if __name__ == '__main__':
    main()



