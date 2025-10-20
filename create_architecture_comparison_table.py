"""
Create comprehensive architecture comparison table with:
1. Detection metrics per class (AR, MR, PR, TR)
2. Classification metrics per class (A4C, PLAX, PSAX)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

# Configuration
THESIS_RESULTS_DIR = Path("yolov5c/thesis results")
CLASSIFICATION_METRICS_DIR = Path("classification_metrics")
OUTPUT_FILE = "ARCHITECTURE_COMPARISON_TABLE.md"

# Model configurations
ARCHITECTURES = {
    'yolov5sc': ['backbone', 'p3', 'p4', 'p5'],
    'yolov5mc': ['backbone', 'p3', 'p4', 'p5'],
    'yolov5mlc': ['backbone', 'p3', 'p4', 'p5']
}

VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5']

# Classes
DETECTION_CLASSES = ['AR', 'MR', 'PR', 'TR']
CLASSIFICATION_CLASSES = ['A4C', 'PLAX', 'PSAX']


def extract_final_epoch_classification_metrics(metrics_file):
    """Extract classification metrics from the last epoch"""
    try:
        df = pd.read_csv(metrics_file, comment='#', skipinitialspace=True)
        # Strip column names of whitespace
        df.columns = df.columns.str.strip()
        
        if len(df) == 0:
            return None
        
        last_row = df.iloc[-1]
        return {
            'cls_accuracy': float(last_row['accuracy']),
            'cls_precision': float(last_row['precision']),
            'cls_recall': float(last_row['recall']),
            'cls_f1': float(last_row['f1_score'])
        }
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None


def extract_detection_metrics(results_file):
    """Extract final detection metrics from results.csv"""
    try:
        df = pd.read_csv(results_file, skipinitialspace=True)
        if len(df) == 0:
            return None
        
        last_row = df.iloc[-1]
        return {
            'det_precision': float(last_row['metrics/precision']),
            'det_recall': float(last_row['metrics/recall']),
            'mAP50': float(last_row['metrics/mAP_0.5']),
            'mAP50_95': float(last_row['metrics/mAP_0.5:0.95'])
        }
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return None


def collect_all_data():
    """Collect all metrics from thesis results"""
    all_data = []
    
    for model_base, configs in ARCHITECTURES.items():
        for config in configs:
            print(f"\nProcessing {model_base}_{config}...")
            
            config_data = {
                'model': model_base,
                'config': config,
                'architecture': f"{model_base}_{config}",
                'versions': []
            }
            
            for version in VERSIONS:
                dir_name = f"{model_base}_{config}_{version}"
                dir_path = THESIS_RESULTS_DIR / dir_name
                
                if not dir_path.exists():
                    print(f"  ⚠️  Directory not found: {dir_path}")
                    continue
                
                print(f"  ✓ Processing {version}...")
                
                version_data = {'version': version}
                
                # Extract classification metrics
                cls_metrics_file = dir_path / "classification_metrics.txt"
                if cls_metrics_file.exists():
                    cls_metrics = extract_final_epoch_classification_metrics(cls_metrics_file)
                    if cls_metrics:
                        version_data.update(cls_metrics)
                
                # Extract detection metrics
                results_file = dir_path / "results.csv"
                if results_file.exists():
                    det_metrics = extract_detection_metrics(results_file)
                    if det_metrics:
                        version_data.update(det_metrics)
                
                if len(version_data) > 1:  # More than just 'version' key
                    config_data['versions'].append(version_data)
            
            if config_data['versions']:
                all_data.append(config_data)
    
    return all_data


def aggregate_versions(config_data):
    """Average metrics across V1-V5"""
    if not config_data['versions']:
        return None
    
    # Collect all metrics
    metrics = {}
    for version_data in config_data['versions']:
        for key, value in version_data.items():
            if key == 'version':
                continue
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    
    # Calculate averages
    aggregated = {
        'architecture': config_data['architecture'],
        'model': config_data['model'],
        'config': config_data['config'],
        'n_versions': len(config_data['versions'])
    }
    
    for key, values in metrics.items():
        aggregated[key] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
    
    return aggregated


def create_markdown_table(aggregated_data):
    """Create comprehensive Markdown table"""
    
    lines = []
    lines.append("# Architecture Performance Comparison")
    lines.append("")
    lines.append("## Complete Performance Table (V1-V5 Averaged)")
    lines.append("")
    lines.append("### Detection + Classification Metrics")
    lines.append("")
    
    # Header
    header = "| Architecture | "
    header += "mAP@0.5 | mAP@0.5:0.95 | Det.Precision | Det.Recall | "
    header += "Cls.Accuracy | Cls.Precision | Cls.Recall | Cls.F1 | "
    header += "Versions |"
    
    separator = "|" + "|".join(["---"] * 10) + "|"
    
    lines.append(header)
    lines.append(separator)
    
    # Sort by model and config
    sorted_data = sorted(aggregated_data, key=lambda x: (x['model'], x['config']))
    
    # Group by model
    current_model = None
    for data in sorted_data:
        if data['model'] != current_model:
            if current_model is not None:
                lines.append(separator)
            current_model = data['model']
        
        row = f"| **{data['architecture']}** | "
        row += f"{data.get('mAP50', 0):.4f} | "
        row += f"{data.get('mAP50_95', 0):.4f} | "
        row += f"{data.get('det_precision', 0):.4f} | "
        row += f"{data.get('det_recall', 0):.4f} | "
        row += f"{data.get('cls_accuracy', 0):.4f} | "
        row += f"{data.get('cls_precision', 0):.4f} | "
        row += f"{data.get('cls_recall', 0):.4f} | "
        row += f"{data.get('cls_f1', 0):.4f} | "
        row += f"{data['n_versions']}/5 |"
        
        lines.append(row)
    
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    
    # Best performers
    lines.append("### 🏆 Best Performers")
    lines.append("")
    
    # Find best in each metric
    best_map50 = max(sorted_data, key=lambda x: x.get('mAP50', 0))
    best_map50_95 = max(sorted_data, key=lambda x: x.get('mAP50_95', 0))
    best_cls_acc = max(sorted_data, key=lambda x: x.get('cls_accuracy', 0))
    best_cls_f1 = max(sorted_data, key=lambda x: x.get('cls_f1', 0))
    
    lines.append(f"- **Best mAP@0.5**: {best_map50['architecture']} ({best_map50.get('mAP50', 0):.4f})")
    lines.append(f"- **Best mAP@0.5:0.95**: {best_map50_95['architecture']} ({best_map50_95.get('mAP50_95', 0):.4f})")
    lines.append(f"- **Best Classification Accuracy**: {best_cls_acc['architecture']} ({best_cls_acc.get('cls_accuracy', 0):.4f})")
    lines.append(f"- **Best Classification F1**: {best_cls_f1['architecture']} ({best_cls_f1.get('cls_f1', 0):.4f})")
    lines.append("")
    
    # Model-wise comparison
    lines.append("### 📊 Model-wise Averages")
    lines.append("")
    lines.append("| Model | Avg mAP@0.5 | Avg Cls.Accuracy | Avg Cls.F1 |")
    lines.append("|-------|-------------|------------------|------------|")
    
    for model_name in ['yolov5sc', 'yolov5mc', 'yolov5mlc']:
        model_data = [d for d in sorted_data if d['model'] == model_name]
        if model_data:
            avg_map50 = np.mean([d.get('mAP50', 0) for d in model_data])
            avg_cls_acc = np.mean([d.get('cls_accuracy', 0) for d in model_data])
            avg_cls_f1 = np.mean([d.get('cls_f1', 0) for d in model_data])
            
            lines.append(f"| **{model_name}** | {avg_map50:.4f} | {avg_cls_acc:.4f} | {avg_cls_f1:.4f} |")
    
    lines.append("")
    
    # Config-wise comparison
    lines.append("### 🔧 Configuration Comparison")
    lines.append("")
    lines.append("| Config | Avg mAP@0.5 | Avg Cls.Accuracy | Count |")
    lines.append("|--------|-------------|------------------|-------|")
    
    for config_name in ['backbone', 'p3', 'p4', 'p5']:
        config_data = [d for d in sorted_data if d['config'] == config_name]
        if config_data:
            avg_map50 = np.mean([d.get('mAP50', 0) for d in config_data])
            avg_cls_acc = np.mean([d.get('cls_accuracy', 0) for d in config_data])
            
            lines.append(f"| **{config_name}** | {avg_map50:.4f} | {avg_cls_acc:.4f} | {len(config_data)} |")
    
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Versions**: Number of dataset versions (V1-V5) successfully processed")
    lines.append("- **Detection Metrics**: Overall detection performance across all classes (AR, MR, PR, TR)")
    lines.append("- **Classification Metrics**: Overall classification performance across all views (A4C, PLAX, PSAX)")
    lines.append("- All values are averaged across V1-V5 datasets")
    lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("⚠️ **Detection per-class metrics (AR, MR, PR, TR)** require running validation.")
    lines.append("⚠️ **Classification per-class metrics (A4C, PLAX, PSAX)** require validation with confusion matrix.")
    lines.append("")
    lines.append("To extract per-class detection metrics, run:")
    lines.append("```bash")
    lines.append("cd yolov5c")
    lines.append("python val.py --weights \"thesis results/yolov5sc_backbone_v1/weights/last.pt\" \\")
    lines.append("    --data \"../Regurgitation-YOLODataset-1/data.yaml\" \\")
    lines.append("    --batch-size 32 --img 416 --task test --verbose")
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("Architecture Comparison Table Generator")
    print("=" * 80)
    
    # Collect all data
    print("\n📊 Collecting data from thesis results...")
    all_data = collect_all_data()
    
    print(f"\n✓ Collected data for {len(all_data)} configurations")
    
    # Aggregate across versions
    print("\n📈 Aggregating metrics across V1-V5...")
    aggregated_data = []
    for config_data in all_data:
        agg = aggregate_versions(config_data)
        if agg:
            aggregated_data.append(agg)
    
    print(f"✓ Aggregated {len(aggregated_data)} configurations")
    
    # Create Markdown table
    print("\n📝 Generating Markdown table...")
    markdown_content = create_markdown_table(aggregated_data)
    
    # Save to file
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n✅ Table saved to: {output_path}")
    print(f"   Total configurations: {len(aggregated_data)}")
    
    # Also save raw data as CSV for further analysis
    csv_file = OUTPUT_FILE.replace('.md', '.csv')
    df = pd.DataFrame(aggregated_data)
    df.to_csv(csv_file, index=False)
    print(f"✅ Raw data saved to: {csv_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

