#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Constraint Violation Comparison Chart
Compare V1 vs V2-V5 datasets constraint violations
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from datetime import datetime

def create_violation_comparison_chart():
    """Create comprehensive violation comparison chart"""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle('YOLOv5WithClassification Dataset Constraint Violations Analysis\nV1 vs V2-V5 Comparison', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Data for comparison
    # V1 data (from ANATOMICAL_CONSTRAINTS_RULES_COMPLETE.md)
    v1_data = {
        'total_files': 1484,
        'violations': 305,
        'violation_rate': 20.55,
        'violation_types': {
            'A4C_PR': 193,
            'A4C_AR': 112,
        }
    }
    
    # V2-V5 data (from our analysis)
    v2_v5_data = {
        'total_files': 1484,  # per dataset
        'violations': 23,     # per dataset
        'violation_rate': 1.55,
        'violation_types': {
            'PLAX_TR': 10,
            'A4C_AR': 4,
            'A4C_PR': 4,
            'PSAX_AR': 3,
            'PSAX_MR': 2,
        }
    }
    
    # 1. Overall Violation Rate Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    datasets = ['V1', 'V2-V5\n(Average)']
    violation_rates = [v1_data['violation_rate'], v2_v5_data['violation_rate']]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars1 = ax1.bar(datasets, violation_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('Constraint Violation Rate Comparison', fontweight='bold')
    
    # Add value labels on bars
    for bar, rate in zip(bars1, violation_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement = (v1_data['violation_rate'] - v2_v5_data['violation_rate']) / v1_data['violation_rate'] * 100
    ax1.text(0.5, max(violation_rates) * 0.7, f'Improvement:\n{improvement:.1f}%', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax1.set_ylim(0, max(violation_rates) * 1.2)
    
    # 2. Violation Count Comparison (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    violation_counts = [v1_data['violations'], v2_v5_data['violations']]
    
    bars2 = ax2.bar(datasets, violation_counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Number of Violations')
    ax2.set_title('Total Violation Count Comparison', fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars2, violation_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylim(0, max(violation_counts) * 1.2)
    
    # 3. V1 Violation Type Distribution (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    v1_types = list(v1_data['violation_types'].keys())
    v1_counts = list(v1_data['violation_types'].values())
    
    wedges, texts, autotexts = ax3.pie(v1_counts, labels=v1_types, autopct='%1.1f%%', 
                                       colors=['#ff9999', '#ffcc99'])
    ax3.set_title('V1 Dataset\nViolation Type Distribution', fontweight='bold')
    
    # 4. V2-V5 Violation Type Distribution (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    v2_types = list(v2_v5_data['violation_types'].keys())
    v2_counts = list(v2_v5_data['violation_types'].values())
    
    wedges, texts, autotexts = ax4.pie(v2_counts, labels=v2_types, autopct='%1.1f%%',
                                       colors=['#99ccff', '#99ffcc', '#ffcc99', '#ff99cc', '#ccccff'])
    ax4.set_title('V2-V5 Datasets\nViolation Type Distribution', fontweight='bold')
    
    # 5. Violation Type Comparison Bar Chart (Bottom Middle & Right)
    ax5 = plt.subplot(2, 3, (5, 6))
    
    # Combine all violation types
    all_types = set(v1_types + v2_types)
    
    # Prepare data for comparison
    v1_comparison = []
    v2_comparison = []
    type_labels = []
    
    for vtype in sorted(all_types):
        type_labels.append(vtype)
        v1_comparison.append(v1_data['violation_types'].get(vtype, 0))
        v2_comparison.append(v2_v5_data['violation_types'].get(vtype, 0))
    
    x = np.arange(len(type_labels))
    width = 0.35
    
    bars_v1 = ax5.bar(x - width/2, v1_comparison, width, label='V1', color='#ff6b6b', alpha=0.8)
    bars_v2 = ax5.bar(x + width/2, v2_comparison, width, label='V2-V5', color='#4ecdc4', alpha=0.8)
    
    ax5.set_xlabel('Violation Type')
    ax5.set_ylabel('Number of Violations')
    ax5.set_title('Violation Type Comparison: V1 vs V2-V5', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(type_labels, rotation=45, ha='right')
    ax5.legend()
    
    # Add value labels on bars
    for bars in [bars_v1, bars_v2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Add summary text box
    summary_text = f"""Summary Statistics:
    
V1 Dataset:
• Total Files: {v1_data['total_files']:,}
• Violations: {v1_data['violations']}
• Rate: {v1_data['violation_rate']:.2f}%

V2-V5 Datasets (each):
• Total Files: {v2_v5_data['total_files']:,}
• Violations: {v2_v5_data['violations']}
• Rate: {v2_v5_data['violation_rate']:.2f}%

Improvement:
• Rate: {improvement:.1f}% better
• Count: {v1_data['violations'] - v2_v5_data['violations']} fewer violations"""
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.90)
    
    # Save the chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'constraint_violations_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Violation comparison chart saved: {filename}")
    
    plt.show()
    return filename

def create_dataset_consistency_chart():
    """Create chart showing V2-V5 consistency"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data for all V2-V5 datasets
    datasets = ['V2', 'V3', 'V4', 'V5']
    violation_files = [23, 23, 23, 23]  # Same for all
    violation_rates = [1.55, 1.55, 1.55, 1.55]  # Same for all
    
    # Colors for each dataset
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    # 1. Violation Files Count
    bars1 = ax1.bar(datasets, violation_files, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Violation Files')
    ax1.set_title('V2-V5 Datasets: Violation Files Count', fontweight='bold')
    ax1.set_ylim(0, 30)
    
    # Add value labels
    for bar, count in zip(bars1, violation_files):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Add consistency annotation
    ax1.text(1.5, 25, 'Perfect\nConsistency!', ha='center', va='center', 
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 2. Violation Rates
    bars2 = ax2.bar(datasets, violation_rates, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Violation Rate (%)')
    ax2.set_title('V2-V5 Datasets: Violation Rates', fontweight='bold')
    ax2.set_ylim(0, 2)
    
    # Add value labels
    for bar, rate in zip(bars2, violation_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('V2-V5 Dataset Consistency Analysis\nAll datasets show identical violation patterns', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'v2_v5_consistency_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Consistency analysis chart saved: {filename}")
    
    plt.show()
    return filename

if __name__ == "__main__":
    print("Creating constraint violation comparison charts...")
    
    # Create main comparison chart
    comparison_file = create_violation_comparison_chart()
    
    # Create consistency chart
    consistency_file = create_dataset_consistency_chart()
    
    print(f"\nCharts created successfully:")
    print(f"1. {comparison_file}")
    print(f"2. {consistency_file}")
    print("\nAnalysis complete!")
