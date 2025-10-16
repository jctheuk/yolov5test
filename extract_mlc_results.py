"""
Extract YOLOv5 MLC training results and create Excel summary
"""
import os
import pandas as pd
import yaml
from pathlib import Path

def extract_results_from_folder(folder_path):
    """Extract key metrics from a training result folder"""
    results = {}
    
    # Extract folder name info
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split('_')
    if len(parts) >= 3:
        results['Architecture'] = parts[1]  # backbone, p3, p4, p5
        results['Dataset'] = parts[2]  # v1, v2, v3, v4, v5
    
    # Read results.csv (last epoch)
    results_csv = os.path.join(folder_path, 'results.csv')
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        if len(df) > 0:
            last_row = df.iloc[-1]
            results['Final_Epoch'] = int(last_row['epoch'])
            results['Train_Box_Loss'] = round(last_row['train/box_loss'], 6)
            results['Train_Obj_Loss'] = round(last_row['train/obj_loss'], 6)
            results['Train_Cls_Loss'] = round(last_row['train/cls_loss'], 6)
            results['Train_Cls_Task_Loss'] = round(last_row['train/cls_task_loss'], 6)
            results['Val_Box_Loss'] = round(last_row['val/box_loss'], 6)
            results['Val_Obj_Loss'] = round(last_row['val/obj_loss'], 6)
            results['Val_Cls_Loss'] = round(last_row['val/cls_loss'], 6)
            results['Val_Cls_Task_Loss'] = round(last_row['val/cls_task_loss'], 6)
            results['Precision'] = round(last_row['metrics/precision'], 6)
            results['Recall'] = round(last_row['metrics/recall'], 6)
            results['mAP_0.5'] = round(last_row['metrics/mAP_0.5'], 6)
            results['mAP_0.5:0.95'] = round(last_row['metrics/mAP_0.5:0.95'], 6)
            
            # Find best epoch
            best_idx = df['metrics/mAP_0.5:0.95'].idxmax()
            best_row = df.iloc[best_idx]
            results['Best_Epoch'] = int(best_row['epoch'])
            results['Best_mAP_0.5:0.95'] = round(best_row['metrics/mAP_0.5:0.95'], 6)
            results['Best_mAP_0.5'] = round(best_row['metrics/mAP_0.5'], 6)
            results['Best_Precision'] = round(best_row['metrics/precision'], 6)
            results['Best_Recall'] = round(best_row['metrics/recall'], 6)
    
    # Read classification_metrics.txt (last epoch)
    cls_metrics = os.path.join(folder_path, 'classification_metrics.txt')
    if os.path.exists(cls_metrics):
        with open(cls_metrics, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 5:
                    results['Cls_Accuracy'] = round(float(last_line[1]), 4)
                    results['Cls_Precision'] = round(float(last_line[2]), 4)
                    results['Cls_Recall'] = round(float(last_line[3]), 4)
                    results['Cls_F1_Score'] = round(float(last_line[4]), 4)
                    
                # Find best classification accuracy
                best_acc = 0
                best_epoch = 0
                for i, line in enumerate(lines[1:], start=0):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        acc = float(parts[1])
                        if acc > best_acc:
                            best_acc = acc
                            best_epoch = i
                
                results['Best_Cls_Epoch'] = best_epoch
                results['Best_Cls_Accuracy'] = round(best_acc, 4)
    
    # Read opt.yaml for configuration
    opt_yaml = os.path.join(folder_path, 'opt.yaml')
    if os.path.exists(opt_yaml):
        with open(opt_yaml, 'r') as f:
            opt = yaml.safe_load(f)
            results['Batch_Size'] = opt.get('batch_size', 'N/A')
            results['Image_Size'] = opt.get('imgsz', 'N/A')
            results['Epochs'] = opt.get('epochs', 'N/A')
            results['Optimizer'] = opt.get('optimizer', 'N/A')
            if 'hyp' in opt and isinstance(opt['hyp'], dict):
                results['LR0'] = opt['hyp'].get('lr0', 'N/A')
                results['Box_Loss_Weight'] = opt['hyp'].get('box', 'N/A')
                results['Cls_Loss_Weight'] = opt['hyp'].get('cls', 'N/A')
                results['Obj_Loss_Weight'] = opt['hyp'].get('obj', 'N/A')
                results['Cls_Task_Weight'] = opt['hyp'].get('cls_task', 'N/A')
    
    return results

def load_thesis_results():
    """Load previous thesis results if available"""
    thesis_files = ['thesis_results_summary.xlsx', 'thesis_results_corrected.xlsx']
    
    for thesis_file in thesis_files:
        if os.path.exists(thesis_file):
            print(f"\nLoading previous thesis results from {thesis_file}...")
            try:
                # Try to read the first sheet
                df = pd.read_excel(thesis_file, sheet_name=0)
                print(f"  - Loaded {len(df)} rows from {thesis_file}")
                return df, thesis_file
            except Exception as e:
                print(f"  - Error loading {thesis_file}: {e}")
                continue
    
    print("  - No previous thesis results found")
    return None, None

def main():
    # Define base directory
    base_dir = Path('yolov5c/runs')
    
    # Define all training folders
    architectures = ['backbone', 'p3', 'p4', 'p5']
    datasets = ['v1', 'v2', 'v3', 'v4', 'v5']
    
    all_results = []
    
    # Iterate through all combinations
    for arch in architectures:
        for ds in datasets:
            folder_name = f'yolov5mlc_{arch}_{ds}'
            folder_path = base_dir / folder_name
            
            if folder_path.exists():
                print(f"Processing {folder_name}...")
                results = extract_results_from_folder(folder_path)
                if results:
                    all_results.append(results)
            else:
                print(f"Warning: {folder_name} not found!")
    
    # Create DataFrame for MLC results
    df = pd.DataFrame(all_results)
    
    # Load previous thesis results
    thesis_df, thesis_file = load_thesis_results()
    
    # Reorder columns for better readability
    column_order = [
        'Architecture', 'Dataset',
        'Final_Epoch', 'Best_Epoch', 'Best_Cls_Epoch',
        # Detection metrics (best)
        'Best_mAP_0.5:0.95', 'Best_mAP_0.5', 'Best_Precision', 'Best_Recall',
        # Classification metrics (best)
        'Best_Cls_Accuracy',
        # Classification metrics (final)
        'Cls_Accuracy', 'Cls_Precision', 'Cls_Recall', 'Cls_F1_Score',
        # Detection metrics (final)
        'Precision', 'Recall', 'mAP_0.5', 'mAP_0.5:0.95',
        # Loss values (final)
        'Train_Box_Loss', 'Train_Obj_Loss', 'Train_Cls_Loss', 'Train_Cls_Task_Loss',
        'Val_Box_Loss', 'Val_Obj_Loss', 'Val_Cls_Loss', 'Val_Cls_Task_Loss',
        # Training config
        'Batch_Size', 'Image_Size', 'Epochs', 'Optimizer', 'LR0',
        'Box_Loss_Weight', 'Cls_Loss_Weight', 'Obj_Loss_Weight', 'Cls_Task_Weight'
    ]
    
    # Reorder columns (keep only existing ones)
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    # Save to Excel
    output_file = 'mlc_training_results_summary.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary sheet for MLC results
        df.to_excel(writer, sheet_name='MLC_Summary', index=False)
        
        # Create separate sheets for each architecture
        for arch in architectures:
            arch_df = df[df['Architecture'] == arch].copy()
            if not arch_df.empty:
                arch_df.to_excel(writer, sheet_name=f'MLC_{arch.upper()}', index=False)
        
        # Create comparison sheet (best metrics only)
        comparison_cols = [
            'Architecture', 'Dataset',
            'Best_Epoch', 'Best_mAP_0.5:0.95', 'Best_mAP_0.5',
            'Best_Precision', 'Best_Recall',
            'Best_Cls_Epoch', 'Best_Cls_Accuracy'
        ]
        comparison_cols = [col for col in comparison_cols if col in df.columns]
        comparison_df = df[comparison_cols].copy()
        comparison_df.to_excel(writer, sheet_name='MLC_Best_Metrics', index=False)
        
        # Add previous thesis results if available
        if thesis_df is not None:
            print(f"\nAdding previous thesis results from {thesis_file}...")
            thesis_df.to_excel(writer, sheet_name='Previous_Thesis_Results', index=False)
            print(f"  - Added {len(thesis_df)} rows to 'Previous_Thesis_Results' sheet")
    
    print(f"\nExcel file created successfully: {output_file}")
    print(f"  - Total MLC experiments: {len(all_results)}")
    print(f"  - Architectures: {', '.join(architectures)}")
    print(f"  - Datasets: {', '.join(datasets)}")
    if thesis_df is not None:
        print(f"  - Previous thesis results: {len(thesis_df)} rows included")
    
    # Print summary statistics
    print("\n=== MLC Training Summary Statistics ===")
    if 'Best_mAP_0.5:0.95' in df.columns:
        best_overall = df.loc[df['Best_mAP_0.5:0.95'].idxmax()]
        print(f"\nBest Detection Performance:")
        print(f"  - Architecture: {best_overall['Architecture']}")
        print(f"  - Dataset: {best_overall['Dataset']}")
        print(f"  - mAP@0.5:0.95: {best_overall['Best_mAP_0.5:0.95']:.4f}")
        print(f"  - mAP@0.5: {best_overall['Best_mAP_0.5']:.4f}")
    
    if 'Best_Cls_Accuracy' in df.columns:
        best_cls = df.loc[df['Best_Cls_Accuracy'].idxmax()]
        print(f"\nBest Classification Performance:")
        print(f"  - Architecture: {best_cls['Architecture']}")
        print(f"  - Dataset: {best_cls['Dataset']}")
        print(f"  - Accuracy: {best_cls['Best_Cls_Accuracy']:.4f}")
    
    # Architecture comparison
    print("\n=== Architecture Comparison (Best mAP@0.5:0.95) ===")
    if 'Best_mAP_0.5:0.95' in df.columns:
        arch_avg = df.groupby('Architecture')['Best_mAP_0.5:0.95'].agg(['mean', 'std', 'max'])
        print(arch_avg.to_string())
    
    return df, thesis_df

if __name__ == '__main__':
    main()
