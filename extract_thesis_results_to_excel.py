#!/usr/bin/env python3
"""
Extract YOLOv5 thesis results to Excel spreadsheet

This script scans all result directories and extracts:
1. Training configurations (model, dataset version, etc.)
2. Final detection metrics (mAP, precision, recall)
3. Final classification metrics (accuracy, precision, recall, f1)
4. Training completion status
"""

import os
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
import re

def parse_experiment_name(exp_name):
    """Parse experiment name to extract configuration details"""
    # Example: yolov5lc_p3_v4 -> model=yolov5lc, pyramid=p3, dataset=v4
    parts = exp_name.split('_')
    
    config = {
        'experiment_name': exp_name,
        'model_size': 'unknown',
        'model_type': 'unknown',
        'pyramid_level': 'unknown',
        'dataset_version': 'unknown'
    }
    
    if len(parts) >= 1:
        model_part = parts[0]
        if 'yolov5lc' in model_part:
            config['model_size'] = 'large'
            config['model_type'] = 'classification'
        elif 'yolov5mc' in model_part:
            config['model_size'] = 'medium' 
            config['model_type'] = 'classification'
        elif 'yolov5sc' in model_part:
            config['model_size'] = 'small'
            config['model_type'] = 'classification'
        elif 'yolov5s' in model_part:
            config['model_size'] = 'small'
            config['model_type'] = 'standard'
        elif 'yolov5m' in model_part:
            config['model_size'] = 'medium'
            config['model_type'] = 'standard'
        elif 'yolov5l' in model_part:
            config['model_size'] = 'large'
            config['model_type'] = 'standard'
        elif 'yolov5x' in model_part:
            config['model_size'] = 'extra_large'
            config['model_type'] = 'standard'
    
    if len(parts) >= 2:
        if 'p3' in parts[1]:
            config['pyramid_level'] = 'P3'
        elif 'p4' in parts[1]:
            config['pyramid_level'] = 'P4'  
        elif 'p5' in parts[1]:
            config['pyramid_level'] = 'P5'
        elif 'backbone' in parts[1]:
            config['pyramid_level'] = 'Backbone'
    
    if len(parts) >= 3:
        version_part = parts[2]
        if version_part.startswith('v'):
            config['dataset_version'] = version_part.upper()
    
    return config

def read_final_detection_metrics(results_csv_path):
    """Read final detection metrics from results.csv"""
    try:
        df = pd.read_csv(results_csv_path)
        if len(df) == 0:
            return {}
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Get the last row (final metrics)
        final_row = df.iloc[-1]
        
        # Helper function to safely get column values
        def safe_get(col_name, default=0):
            if col_name in df.columns:
                return float(final_row.get(col_name, default))
            return default
        
        metrics = {
            'final_epoch': int(final_row.get('epoch', -1)),
            'train_box_loss': safe_get('train/box_loss'),
            'train_obj_loss': safe_get('train/obj_loss'), 
            'train_cls_loss': safe_get('train/cls_loss'),
            'train_cls_task_loss': safe_get('train/cls_task_loss'),
            'val_box_loss': safe_get('val/box_loss'),
            'val_obj_loss': safe_get('val/obj_loss'),
            'val_cls_loss': safe_get('val/cls_loss'),
            'val_cls_task_loss': safe_get('val/cls_task_loss'),
            'precision': safe_get('metrics/precision'),
            'recall': safe_get('metrics/recall'),
            'mAP_50': safe_get('metrics/mAP_0.5'),
            'mAP_50_95': safe_get('metrics/mAP_0.5:0.95')
        }
        
        # Find best mAP values if columns exist
        if 'metrics/mAP_0.5' in df.columns and not df['metrics/mAP_0.5'].isna().all():
            best_map_50_idx = df['metrics/mAP_0.5'].idxmax()
            metrics.update({
                'best_mAP_50': float(df.loc[best_map_50_idx, 'metrics/mAP_0.5']),
                'best_mAP_50_epoch': int(df.loc[best_map_50_idx, 'epoch'])
            })
        else:
            metrics.update({'best_mAP_50': 0, 'best_mAP_50_epoch': -1})
            
        if 'metrics/mAP_0.5:0.95' in df.columns and not df['metrics/mAP_0.5:0.95'].isna().all():
            best_map_50_95_idx = df['metrics/mAP_0.5:0.95'].idxmax()
            metrics.update({
                'best_mAP_50_95': float(df.loc[best_map_50_95_idx, 'metrics/mAP_0.5:0.95']),
                'best_mAP_50_95_epoch': int(df.loc[best_map_50_95_idx, 'epoch'])
            })
        else:
            metrics.update({'best_mAP_50_95': 0, 'best_mAP_50_95_epoch': -1})
        
        return metrics
    except Exception as e:
        print(f"Error reading {results_csv_path}: {e}")
        return {}

def read_final_classification_metrics(class_metrics_path):
    """Read final classification metrics from classification_metrics.txt"""
    try:
        # Read file, manually specifying column names since the header is commented
        column_names = ['epoch', 'accuracy', 'precision', 'recall', 'f1_score']
        df = pd.read_csv(class_metrics_path, comment='#', names=column_names)
        
        if len(df) == 0:
            return {}
        
        # Get the last row (final metrics)
        final_row = df.iloc[-1]
        
        metrics = {
            'cls_final_epoch': int(final_row.get('epoch', -1)),
            'cls_final_accuracy': float(final_row.get('accuracy', 0)),
            'cls_final_precision': float(final_row.get('precision', 0)),
            'cls_final_recall': float(final_row.get('recall', 0)),
            'cls_final_f1': float(final_row.get('f1_score', 0))
        }
        
        # Find best accuracy
        if 'accuracy' in df.columns and not df['accuracy'].isna().all():
            best_acc_idx = df['accuracy'].idxmax()
            metrics.update({
                'cls_best_accuracy': float(df.loc[best_acc_idx, 'accuracy']),
                'cls_best_accuracy_epoch': int(df.loc[best_acc_idx, 'epoch']),
                'cls_best_f1': float(df.loc[best_acc_idx, 'f1_score'])
            })
        else:
            metrics.update({
                'cls_best_accuracy': 0,
                'cls_best_accuracy_epoch': -1,
                'cls_best_f1': 0
            })
        
        return metrics
    except Exception as e:
        print(f"Error reading {class_metrics_path}: {e}")
        return {}

def read_training_config(opt_yaml_path, hyp_yaml_path):
    """Read training configuration from opt.yaml and hyp.yaml"""
    config = {}
    
    try:
        if os.path.exists(opt_yaml_path):
            with open(opt_yaml_path, 'r', encoding='utf-8') as f:
                opt_data = yaml.safe_load(f)
                config.update({
                    'batch_size': opt_data.get('batch_size', 'unknown'),
                    'epochs': opt_data.get('epochs', 'unknown'),
                    'img_size': opt_data.get('imgsz', 'unknown'),
                    'data_yaml': opt_data.get('data', 'unknown'),
                    'weights': opt_data.get('weights', 'unknown')
                })
    except Exception as e:
        print(f"Error reading {opt_yaml_path}: {e}")
    
    try:
        if os.path.exists(hyp_yaml_path):
            with open(hyp_yaml_path, 'r', encoding='utf-8') as f:
                hyp_data = yaml.safe_load(f)
                config.update({
                    'lr0': hyp_data.get('lr0', 'unknown'),
                    'momentum': hyp_data.get('momentum', 'unknown'),
                    'weight_decay': hyp_data.get('weight_decay', 'unknown'),
                    'classification_enabled': hyp_data.get('classification_enabled', 'unknown'),
                    'cls_loss_weight': hyp_data.get('cls_loss_weight', 'unknown')
                })
    except Exception as e:
        print(f"Error reading {hyp_yaml_path}: {e}")
    
    return config

def scan_thesis_results(results_dir):
    """Scan all experiment directories and extract data"""
    results_data = []
    
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        
        if not os.path.isdir(exp_path):
            continue
            
        print(f"Processing: {exp_dir}")
        
        # Parse experiment configuration
        config = parse_experiment_name(exp_dir)
        
        # Check if training completed successfully
        results_csv = os.path.join(exp_path, 'results.csv')
        class_metrics = os.path.join(exp_path, 'classification_metrics.txt')
        opt_yaml = os.path.join(exp_path, 'opt.yaml')
        hyp_yaml = os.path.join(exp_path, 'hyp.yaml')
        weights_dir = os.path.join(exp_path, 'weights')
        
        # Determine completion status
        has_results = os.path.exists(results_csv)
        has_class_metrics = os.path.exists(class_metrics)
        has_weights = os.path.exists(weights_dir) and len(os.listdir(weights_dir)) > 0
        has_config = os.path.exists(opt_yaml)
        
        if has_results and has_class_metrics and has_weights:
            status = 'Complete'
        elif has_weights:
            status = 'Partial'
        else:
            status = 'Failed'
        
        # Initialize row data
        row_data = {
            'Status': status,
            **config
        }
        
        # Extract metrics if available
        if has_results:
            detection_metrics = read_final_detection_metrics(results_csv)
            row_data.update(detection_metrics)
        
        if has_class_metrics:
            classification_metrics = read_final_classification_metrics(class_metrics)
            row_data.update(classification_metrics)
            
        if has_config:
            training_config = read_training_config(opt_yaml, hyp_yaml)
            row_data.update(training_config)
        
        results_data.append(row_data)
        
    return results_data

def create_excel_summary(results_data, output_file):
    """Create comprehensive Excel file with multiple sheets"""
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Main summary sheet
        df_main = pd.DataFrame(results_data)
        
        # Reorder columns for better readability
        column_order = [
            'experiment_name', 'Status', 'model_size', 'model_type', 'pyramid_level', 'dataset_version',
            'final_epoch', 'best_mAP_50', 'best_mAP_50_epoch', 'best_mAP_50_95', 'best_mAP_50_95_epoch',
            'cls_best_accuracy', 'cls_best_accuracy_epoch', 'cls_best_f1',
            'precision', 'recall', 'mAP_50', 'mAP_50_95',
            'cls_final_accuracy', 'cls_final_precision', 'cls_final_recall', 'cls_final_f1',
            'batch_size', 'epochs', 'img_size', 'lr0', 'classification_enabled', 'cls_loss_weight'
        ]
        
        # Keep only existing columns
        available_columns = [col for col in column_order if col in df_main.columns]
        remaining_columns = [col for col in df_main.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        df_main_ordered = df_main[final_columns] if final_columns else df_main
        df_main_ordered.to_excel(writer, sheet_name='Main Summary', index=False)
        
        # Complete results only
        df_complete = df_main[df_main['Status'] == 'Complete'] if 'Status' in df_main.columns else df_main
        if not df_complete.empty:
            df_complete.to_excel(writer, sheet_name='Complete Results', index=False)
        
        # Performance comparison
        if not df_complete.empty and 'best_mAP_50' in df_complete.columns:
            performance_cols = [
                'experiment_name', 'model_size', 'pyramid_level', 'dataset_version',
                'best_mAP_50', 'best_mAP_50_95', 'cls_best_accuracy', 'cls_best_f1'
            ]
            available_perf_cols = [col for col in performance_cols if col in df_complete.columns]
            df_performance = df_complete[available_perf_cols].sort_values('best_mAP_50', ascending=False)
            df_performance.to_excel(writer, sheet_name='Performance Ranking', index=False)
        
        # Configuration summary  
        config_cols = [
            'experiment_name', 'model_size', 'model_type', 'pyramid_level', 'dataset_version',
            'batch_size', 'epochs', 'img_size', 'lr0', 'classification_enabled'
        ]
        available_config_cols = [col for col in config_cols if col in df_main.columns]
        if available_config_cols:
            df_config = df_main[available_config_cols]
            df_config.to_excel(writer, sheet_name='Configurations', index=False)

def main():
    results_dir = "yolov5c/thesis results"
    output_file = "thesis_results_corrected.xlsx"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print("Scanning thesis results...")
    results_data = scan_thesis_results(results_dir)
    
    print(f"Found {len(results_data)} experiment directories")
    
    # Display summary
    complete_count = sum(1 for r in results_data if r.get('Status') == 'Complete')
    partial_count = sum(1 for r in results_data if r.get('Status') == 'Partial')
    failed_count = sum(1 for r in results_data if r.get('Status') == 'Failed')
    
    print(f"Complete experiments: {complete_count}")
    print(f"Partial experiments: {partial_count}")
    print(f"Failed experiments: {failed_count}")
    
    print(f"Creating Excel file: {output_file}")
    create_excel_summary(results_data, output_file)
    
    print(f"[SUCCESS] Excel file created successfully: {output_file}")
    
    # Show some quick stats
    if complete_count > 0:
        complete_results = [r for r in results_data if r.get('Status') == 'Complete']
        if complete_results and 'best_mAP_50' in complete_results[0]:
            best_map_results = [r for r in complete_results if 'best_mAP_50' in r and r['best_mAP_50']]
            if best_map_results:
                best_experiment = max(best_map_results, key=lambda x: x.get('best_mAP_50', 0))
                print(f"\n[BEST mAP] Best mAP@0.5: {best_experiment.get('best_mAP_50', 'N/A'):.4f} ({best_experiment.get('experiment_name', 'Unknown')})")
        
        if 'cls_best_accuracy' in complete_results[0]:
            best_acc_results = [r for r in complete_results if 'cls_best_accuracy' in r and r['cls_best_accuracy']]
            if best_acc_results:
                best_cls_experiment = max(best_acc_results, key=lambda x: x.get('cls_best_accuracy', 0))
                print(f"[BEST ACC] Best Classification Accuracy: {best_cls_experiment.get('cls_best_accuracy', 'N/A'):.4f} ({best_cls_experiment.get('experiment_name', 'Unknown')})")

if __name__ == "__main__":
    main()
