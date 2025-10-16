#!/usr/bin/env python3
"""
Quick analysis of thesis results Excel file
"""

import pandas as pd
import numpy as np

def analyze_results():
    try:
        # Load the Excel file
        excel_file = "thesis_results_corrected.xlsx"
        
        # Read the main summary sheet
        df = pd.read_excel(excel_file, sheet_name='Main Summary')
        
        print("=" * 80)
        print("THESIS RESULTS ANALYSIS")
        print("=" * 80)
        
        # Basic stats
        total_experiments = len(df)
        complete_experiments = len(df[df['Status'] == 'Complete'])
        
        print(f"\nOverall Statistics:")
        print(f"  Total Experiments: {total_experiments}")
        print(f"  Complete Experiments: {complete_experiments}")
        print(f"  Success Rate: {complete_experiments/total_experiments*100:.1f}%")
        
        # Filter complete results for analysis
        complete_df = df[df['Status'] == 'Complete'].copy()
        
        if len(complete_df) > 0:
            print(f"\n" + "=" * 50)
            print("DETECTION PERFORMANCE RANKING")
            print("=" * 50)
            
            # Sort by best mAP@0.5
            if 'best_mAP_50' in complete_df.columns:
                mAP_ranking = complete_df.nlargest(10, 'best_mAP_50')[
                    ['experiment_name', 'model_size', 'pyramid_level', 'dataset_version', 'best_mAP_50', 'best_mAP_50_95']
                ].round(4)
                
                print("\nTop 10 Models by mAP@0.5:")
                for i, (_, row) in enumerate(mAP_ranking.iterrows(), 1):
                    print(f"{i:2d}. {row['experiment_name']:20s} | "
                          f"mAP@0.5: {row['best_mAP_50']:.4f} | "
                          f"mAP@0.5:0.95: {row['best_mAP_50_95']:.4f} | "
                          f"{row['model_size']}-{row['pyramid_level']}-{row['dataset_version']}")
            
            print(f"\n" + "=" * 50)
            print("CLASSIFICATION PERFORMANCE RANKING")
            print("=" * 50)
            
            # Sort by classification accuracy
            if 'cls_best_accuracy' in complete_df.columns:
                cls_ranking = complete_df.nlargest(10, 'cls_best_accuracy')[
                    ['experiment_name', 'model_size', 'pyramid_level', 'dataset_version', 'cls_best_accuracy', 'cls_best_f1']
                ].round(4)
                
                print("\nTop 10 Models by Classification Accuracy:")
                for i, (_, row) in enumerate(cls_ranking.iterrows(), 1):
                    print(f"{i:2d}. {row['experiment_name']:20s} | "
                          f"Accuracy: {row['cls_best_accuracy']:.4f} | "
                          f"F1: {row['cls_best_f1']:.4f} | "
                          f"{row['model_size']}-{row['pyramid_level']}-{row['dataset_version']}")
            
            print(f"\n" + "=" * 50)
            print("MODEL SIZE COMPARISON")
            print("=" * 50)
            
            # Group by model size
            if 'model_size' in complete_df.columns and 'best_mAP_50' in complete_df.columns:
                size_stats = complete_df.groupby('model_size').agg({
                    'best_mAP_50': ['count', 'mean', 'max'],
                    'cls_best_accuracy': ['mean', 'max'] if 'cls_best_accuracy' in complete_df.columns else ['count']
                }).round(4)
                
                print("\nPerformance by Model Size:")
                print(f"{'Model Size':<15s} | {'Count':<5s} | {'Avg mAP':<8s} | {'Max mAP':<8s} | {'Avg Acc':<8s} | {'Max Acc':<8s}")
                print("-" * 70)
                
                for model_size in size_stats.index:
                    count = int(size_stats.loc[model_size, ('best_mAP_50', 'count')])
                    avg_map = size_stats.loc[model_size, ('best_mAP_50', 'mean')]
                    max_map = size_stats.loc[model_size, ('best_mAP_50', 'max')]
                    
                    if 'cls_best_accuracy' in complete_df.columns:
                        avg_acc = size_stats.loc[model_size, ('cls_best_accuracy', 'mean')]
                        max_acc = size_stats.loc[model_size, ('cls_best_accuracy', 'max')]
                        print(f"{model_size:<15s} | {count:<5d} | {avg_map:<8.4f} | {max_map:<8.4f} | {avg_acc:<8.4f} | {max_acc:<8.4f}")
                    else:
                        print(f"{model_size:<15s} | {count:<5d} | {avg_map:<8.4f} | {max_map:<8.4f} | {'N/A':<8s} | {'N/A':<8s}")
            
            print(f"\n" + "=" * 50)
            print("DATASET VERSION COMPARISON")
            print("=" * 50)
            
            # Group by dataset version
            if 'dataset_version' in complete_df.columns and 'best_mAP_50' in complete_df.columns:
                version_stats = complete_df.groupby('dataset_version').agg({
                    'best_mAP_50': ['count', 'mean', 'max'],
                    'cls_best_accuracy': ['mean', 'max'] if 'cls_best_accuracy' in complete_df.columns else ['count']
                }).round(4)
                
                print("\nPerformance by Dataset Version:")
                print(f"{'Dataset':<10s} | {'Count':<5s} | {'Avg mAP':<8s} | {'Max mAP':<8s} | {'Avg Acc':<8s} | {'Max Acc':<8s}")
                print("-" * 60)
                
                for version in sorted(version_stats.index):
                    count = int(version_stats.loc[version, ('best_mAP_50', 'count')])
                    avg_map = version_stats.loc[version, ('best_mAP_50', 'mean')]
                    max_map = version_stats.loc[version, ('best_mAP_50', 'max')]
                    
                    if 'cls_best_accuracy' in complete_df.columns:
                        avg_acc = version_stats.loc[version, ('cls_best_accuracy', 'mean')]
                        max_acc = version_stats.loc[version, ('cls_best_accuracy', 'max')]
                        print(f"{version:<10s} | {count:<5d} | {avg_map:<8.4f} | {max_map:<8.4f} | {avg_acc:<8.4f} | {max_acc:<8.4f}")
                    else:
                        print(f"{version:<10s} | {count:<5d} | {avg_map:<8.4f} | {max_map:<8.4f} | {'N/A':<8s} | {'N/A':<8s}")
            
            print(f"\n" + "=" * 50)
            print("BEST OVERALL COMBINATIONS")
            print("=" * 50)
            
            if 'best_mAP_50' in complete_df.columns and 'cls_best_accuracy' in complete_df.columns:
                # Create combined score (you can adjust weights as needed)
                complete_df['combined_score'] = (complete_df['best_mAP_50'] * 0.6 + complete_df['cls_best_accuracy'] * 0.4)
                
                best_combined = complete_df.nlargest(5, 'combined_score')[
                    ['experiment_name', 'model_size', 'pyramid_level', 'dataset_version', 
                     'best_mAP_50', 'cls_best_accuracy', 'combined_score']
                ].round(4)
                
                print("\nTop 5 Models by Combined Score (60% mAP + 40% Accuracy):")
                for i, (_, row) in enumerate(best_combined.iterrows(), 1):
                    print(f"{i}. {row['experiment_name']:20s} | "
                          f"Score: {row['combined_score']:.4f} | "
                          f"mAP: {row['best_mAP_50']:.4f} | "
                          f"Acc: {row['cls_best_accuracy']:.4f}")
        
        print(f"\n" + "=" * 80)
        print(f"Analysis complete! Check '{excel_file}' for detailed data.")
        print("The Excel file contains multiple sheets:")
        print("  - Main Summary: All experiments with full details")
        print("  - Complete Results: Only successfully completed experiments")
        print("  - Performance Ranking: Sorted by performance metrics")
        print("  - Configurations: Training configuration summary")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        print("Make sure 'thesis_results_summary.xlsx' exists and is accessible.")

if __name__ == "__main__":
    analyze_results()
