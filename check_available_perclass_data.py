"""
Check what per-class data is available in existing results.
"""

import os
import pandas as pd
from pathlib import Path

def check_thesis_results():
    """Check thesis results directory for per-class data."""
    thesis_results_dir = "yolov5c/thesis results"
    
    # Check a few sample folders
    samples = [
        'yolov5mc_backbone_v1',
        'yolov5sc_p3_v1',
        'yolov5mlc_p5_v1'
    ]
    
    for sample in samples:
        sample_path = os.path.join(thesis_results_dir, sample)
        if not os.path.exists(sample_path):
            continue
        
        print(f"\n{'='*80}")
        print(f"Checking: {sample}")
        print('='*80)
        
        # List files
        files = os.listdir(sample_path)
        print(f"\nFiles in directory:")
        for f in sorted(files):
            print(f"  - {f}")
        
        # Check classification_metrics.txt
        cls_metrics = os.path.join(sample_path, 'classification_metrics.txt')
        if os.path.exists(cls_metrics):
            print(f"\nClassification metrics (last 50 lines):")
            with open(cls_metrics, 'r') as f:
                lines = f.readlines()
                print(''.join(lines[-50:]))
        
        # Check results.csv
        results_csv = os.path.join(sample_path, 'results.csv')
        if os.path.exists(results_csv):
            print(f"\nResults CSV columns:")
            df = pd.read_csv(results_csv)
            print(df.columns.tolist())
            print(f"\nLast row:")
            print(df.iloc[-1])


def check_yolov5original():
    """Check yolov5original results."""
    original_dir = "yolov5original/runs/train-cls"
    
    if not os.path.exists(original_dir):
        print(f"\n{original_dir} not found")
        return
    
    # Check sample folders
    samples = ['classifys_v1', 'classifym_v1', 'classifyl_v1']
    
    for sample in samples:
        sample_path = os.path.join(original_dir, sample)
        if not os.path.exists(sample_path):
            continue
        
        print(f"\n{'='*80}")
        print(f"Checking: {sample}")
        print('='*80)
        
        # List files
        files = os.listdir(sample_path)
        print(f"\nFiles in directory:")
        for f in sorted(files):
            if not f.startswith('.'):
                print(f"  - {f}")
        
        # Check for detailed_metrics.csv
        detailed_metrics = os.path.join(sample_path, 'detailed_metrics.csv')
        if os.path.exists(detailed_metrics):
            print(f"\nDetailed metrics CSV:")
            df = pd.read_csv(detailed_metrics)
            print(df.to_string(index=False))


def check_val_cls_results():
    """Check val-cls results for per-class data."""
    val_dir = "yolov5original/runs/val-cls"
    
    if not os.path.exists(val_dir):
        print(f"\n{val_dir} not found")
        return
    
    # List directories
    dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    print(f"\n{'='*80}")
    print(f"Val-cls directories ({len(dirs)} found):")
    print('='*80)
    for d in sorted(dirs)[:10]:
        print(f"  - {d}")
    
    # Check first few
    for d in sorted(dirs)[:3]:
        d_path = os.path.join(val_dir, d)
        files = os.listdir(d_path)
        print(f"\n{d} files:")
        for f in sorted(files):
            print(f"  - {f}")
            
            # If it's detailed_metrics.csv, show it
            if f == 'detailed_metrics.csv':
                csv_path = os.path.join(d_path, f)
                df = pd.read_csv(csv_path)
                print(f"\n  Content:")
                print(df.to_string(index=False))


def main():
    print("Checking available per-class data in existing results...")
    
    check_thesis_results()
    check_yolov5original()
    check_val_cls_results()
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print("""
To get complete per-class metrics, you have two options:

1. FAST: Use existing data where available
   - YOLOv5 Original has per-class classification in detailed_metrics.csv
   - YOLOv5c has overall metrics only in current thesis results
   
2. COMPLETE: Run validation to extract per-class metrics
   - Run: python extract_perclass_metrics.py
   - This will take 30-60 minutes
   - Will generate per-class detection AND classification metrics
    """)


if __name__ == '__main__':
    main()


