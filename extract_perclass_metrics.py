"""
Extract per-class metrics from trained models by running validation.

This script:
1. Finds all trained model weights
2. Runs validation with detailed metrics
3. Extracts per-class detection (AP, AR per class)
4. Extracts per-class classification (Acc, Prec, Recall, F1 per class)
5. Aggregates v1-v5 and generates detailed output
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add yolov5c to path
sys.path.insert(0, os.path.join(os.getcwd(), 'yolov5c'))

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


def find_model_weight(base_path, architecture, loss_type, version):
    """Find the best.pt weight file for a model."""
    # Pattern: yolov5c/thesis results/yolov5{loss_type}_{architecture}_{version}/weights/best.pt
    pattern = f"{base_path}/yolov5c/thesis results/yolov5{loss_type}_{architecture}_{version}/weights/best.pt"
    
    if os.path.exists(pattern):
        return pattern
    
    # Try last.pt if best.pt doesn't exist
    pattern_last = pattern.replace('best.pt', 'last.pt')
    if os.path.exists(pattern_last):
        return pattern_last
    
    return None


def get_dataset_path(version):
    """Get dataset path for a version."""
    version_map = {
        'v1': 'regurgitationV1',
        'v2': 'regurgitationV2',
        'v3': 'regurgitationV3',
        'v4': 'regurgitationV4',
        'v5': 'regurgitationV5',
    }
    
    dataset_name = version_map.get(version, 'regurgitationV1')
    data_yaml = f"{dataset_name}/data.yaml"
    
    if os.path.exists(data_yaml):
        return data_yaml
    return None


def run_validation_for_perclass(weight_path, data_yaml, output_dir):
    """Run validation and extract per-class metrics."""
    if not os.path.exists(weight_path):
        print(f"  Weight not found: {weight_path}")
        return None
    
    if not os.path.exists(data_yaml):
        print(f"  Data YAML not found: {data_yaml}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run validation with verbose output
    cmd = [
        'python', 'yolov5c/val.py',
        '--weights', weight_path,
        '--data', data_yaml,
        '--batch-size', '32',
        '--img', '640',
        '--task', 'test',
        '--save-txt',
        '--save-conf',
        '--verbose',
        '--project', output_dir,
        '--name', 'val',
        '--exist-ok'
    ]
    
    print(f"  Running validation...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Parse output for per-class metrics
        output = result.stdout + result.stderr
        
        # Look for per-class AP lines
        perclass_metrics = {
            'detection': {},
            'classification': {}
        }
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # Detection per-class: "Class     Images  Instances          P          R      mAP50   mAP50-95:"
            if 'all' in line and 'Instances' in lines[i-1]:
                # Parse class metrics
                for j in range(i-2, max(0, i-10), -1):
                    parts = lines[j].split()
                    if len(parts) >= 6 and parts[0] in ['0', '1']:
                        class_idx = int(parts[0])
                        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class{class_idx}"
                        perclass_metrics['detection'][class_name] = {
                            'images': int(parts[1]),
                            'instances': int(parts[2]),
                            'precision': float(parts[3]),
                            'recall': float(parts[4]),
                            'mAP50': float(parts[5]),
                            'mAP50-95': float(parts[6]) if len(parts) > 6 else None
                        }
        
        # Look for classification metrics
        if 'Classification Accuracy:' in output or 'Overall Accuracy:' in output:
            for line in lines:
                if 'Class:' in line:
                    class_name = line.split(':')[-1].strip()
                    # Next lines have metrics
                    idx = lines.index(line)
                    metrics = {}
                    for k in range(1, 5):
                        if idx + k < len(lines) and ':' in lines[idx + k]:
                            metric_line = lines[idx + k]
                            metric_name, metric_value = metric_line.split(':', 1)
                            metric_name = metric_name.strip()
                            try:
                                metric_value = float(metric_value.strip().replace('%', ''))
                                metrics[metric_name] = metric_value
                            except:
                                pass
                    
                    if metrics:
                        perclass_metrics['classification'][class_name] = metrics
        
        return perclass_metrics
    
    except subprocess.TimeoutExpired:
        print(f"  Validation timed out")
        return None
    except Exception as e:
        print(f"  Validation error: {e}")
        return None


def extract_all_perclass_metrics(base_path):
    """Extract per-class metrics for all models."""
    all_metrics = {}
    
    for architecture, loss_type in YOLOV5C_MODELS:
        model_key = f"yolov5{loss_type}_{architecture}"
        print(f"\nProcessing {model_key}...")
        
        version_metrics = []
        
        for version in VERSIONS:
            print(f"  Version {version}...")
            
            weight_path = find_model_weight(base_path, architecture, loss_type, version)
            if not weight_path:
                print(f"    Weight not found, skipping")
                continue
            
            data_yaml = get_dataset_path(version)
            if not data_yaml:
                print(f"    Data YAML not found, skipping")
                continue
            
            output_dir = f"perclass_metrics_temp/{model_key}_{version}"
            
            metrics = run_validation_for_perclass(weight_path, data_yaml, output_dir)
            if metrics:
                version_metrics.append(metrics)
        
        # Aggregate v1-v5
        if version_metrics:
            aggregated = aggregate_perclass_metrics(version_metrics)
            all_metrics[model_key] = aggregated
    
    return all_metrics


def aggregate_perclass_metrics(version_metrics):
    """Aggregate per-class metrics across versions."""
    aggregated = {
        'detection': {},
        'classification': {}
    }
    
    # Aggregate detection metrics
    all_classes = set()
    for vm in version_metrics:
        all_classes.update(vm['detection'].keys())
    
    for class_name in all_classes:
        class_metrics = {}
        for metric_name in ['precision', 'recall', 'mAP50', 'mAP50-95']:
            values = []
            for vm in version_metrics:
                if class_name in vm['detection'] and metric_name in vm['detection'][class_name]:
                    val = vm['detection'][class_name][metric_name]
                    if val is not None:
                        values.append(val)
            
            if values:
                class_metrics[metric_name] = np.mean(values)
        
        if class_metrics:
            aggregated['detection'][class_name] = class_metrics
    
    # Aggregate classification metrics
    all_classes = set()
    for vm in version_metrics:
        all_classes.update(vm['classification'].keys())
    
    for class_name in all_classes:
        class_metrics = {}
        for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            values = []
            for vm in version_metrics:
                if class_name in vm['classification'] and metric_name in vm['classification'][class_name]:
                    values.append(vm['classification'][class_name][metric_name])
            
            if values:
                class_metrics[metric_name] = np.mean(values)
        
        if class_metrics:
            aggregated['classification'][class_name] = class_metrics
    
    return aggregated


def save_perclass_results(all_metrics, output_file):
    """Save per-class metrics to JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nSaved per-class metrics to {output_file}")


def create_perclass_tables(all_metrics):
    """Create detailed tables with per-class metrics."""
    
    # Detection per-class table
    detection_rows = []
    for model_key, metrics in all_metrics.items():
        for class_name, class_metrics in metrics['detection'].items():
            row = {
                'Model': model_key,
                'Class': class_name,
                'Precision': class_metrics.get('precision'),
                'Recall': class_metrics.get('recall'),
                'mAP@0.5': class_metrics.get('mAP50'),
                'mAP@0.5:0.95': class_metrics.get('mAP50-95')
            }
            detection_rows.append(row)
    
    detection_df = pd.DataFrame(detection_rows)
    detection_df.to_csv('results/perclass_detection_metrics.csv', index=False)
    print("Saved: results/perclass_detection_metrics.csv")
    
    # Classification per-class table
    classification_rows = []
    for model_key, metrics in all_metrics.items():
        for class_name, class_metrics in metrics['classification'].items():
            row = {
                'Model': model_key,
                'Class': class_name,
                'Accuracy': class_metrics.get('Accuracy'),
                'Precision': class_metrics.get('Precision'),
                'Recall': class_metrics.get('Recall'),
                'F1-Score': class_metrics.get('F1-Score')
            }
            classification_rows.append(row)
    
    classification_df = pd.DataFrame(classification_rows)
    classification_df.to_csv('results/perclass_classification_metrics.csv', index=False)
    print("Saved: results/perclass_classification_metrics.csv")
    
    return detection_df, classification_df


def main():
    base_path = os.getcwd()
    
    print("=" * 80)
    print("Extracting per-class metrics from all models...")
    print("This will run validation on all model weights.")
    print("This may take a long time (30+ minutes).")
    print("=" * 80)
    
    # Extract metrics
    all_metrics = extract_all_perclass_metrics(base_path)
    
    # Save to JSON
    save_perclass_results(all_metrics, 'results/perclass_metrics_detailed.json')
    
    # Create CSV tables
    detection_df, classification_df = create_perclass_tables(all_metrics)
    
    print("\n" + "=" * 80)
    print("Per-class metrics extraction complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. results/perclass_metrics_detailed.json")
    print("  2. results/perclass_detection_metrics.csv")
    print("  3. results/perclass_classification_metrics.csv")
    
    print("\nDetection per-class preview:")
    print(detection_df.head(10).to_string(index=False))
    
    print("\nClassification per-class preview:")
    print(classification_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()


