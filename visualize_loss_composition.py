#!/usr/bin/env python3
"""
Visualize Loss Composition for Different Hyperparameter Configurations

This script visualizes how different hyperparameter settings affect the loss composition
in YOLOv5WithClassification training with the improved loss scaling strategy (Plan B).

Loss Scaling Strategy (Plan B):
- Detection losses: (lbox + lobj + lcls) * batch_size  (mean → scaled)
- Classification loss: cls_task (mean → not scaled)
- Constraint loss: lconstraint (sum → not scaled)
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

# Configuration for loss calculation
BATCH_SIZE = 16

# Typical loss values (before scaling)
TYPICAL_LOSSES = {
    'lbox': 0.05,      # Box loss (mean)
    'lobj': 0.03,      # Objectness loss (mean)
    'lcls_det': 0.02,  # Detection class loss (mean)
    'lcls_task_base': 1.2,  # Base classification loss (mean, before cls_task weight)
    'lconstraint_per_sample': 0.03,  # Constraint penalty per sample
}


def calculate_loss_composition(hyp, batch_size=16):
    """
    Calculate loss composition based on hyperparameters.
    
    Args:
        hyp: Dictionary of hyperparameters
        batch_size: Batch size for training
        
    Returns:
        Dictionary with loss components and statistics
    """
    # Get hyperparameters
    box_gain = hyp.get('box', 0.05)
    obj_gain = hyp.get('obj', 1.0)
    cls_gain = hyp.get('cls', 0.5)
    cls_task = hyp.get('cls_task', 0.3)
    constraint_weight = hyp.get('constraint_weight', 0.3)
    use_constraints = hyp.get('use_anatomical_constraints', True)
    
    # Calculate individual losses
    lbox = TYPICAL_LOSSES['lbox'] * box_gain
    lobj = TYPICAL_LOSSES['lobj'] * obj_gain
    lcls_det = TYPICAL_LOSSES['lcls_det'] * cls_gain
    
    # Detection loss (scaled by batch_size)
    detection_loss = (lbox + lobj + lcls_det) * batch_size
    
    # Classification loss (not scaled)
    classification_loss = TYPICAL_LOSSES['lcls_task_base'] * cls_task
    
    # Constraint loss (sum, not scaled)
    if use_constraints:
        lconstraint = TYPICAL_LOSSES['lconstraint_per_sample'] * batch_size * constraint_weight
        constraint_loss = lconstraint
    else:
        constraint_loss = 0.0
    
    # Total loss
    total_loss = detection_loss + classification_loss + constraint_loss
    
    return {
        'detection': detection_loss,
        'classification': classification_loss,
        'constraint': constraint_loss,
        'total': total_loss,
        'detection_pct': (detection_loss / total_loss) * 100,
        'classification_pct': (classification_loss / total_loss) * 100,
        'constraint_pct': (constraint_loss / total_loss) * 100,
        # Individual components
        'lbox': lbox * batch_size,
        'lobj': lobj * batch_size,
        'lcls_det': lcls_det * batch_size,
    }


def load_hyp(hyp_path):
    """Load hyperparameters from YAML file."""
    with open(hyp_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_loss_visualization(configs, output_path='loss_composition_comparison.png'):
    """
    Create comprehensive visualization of loss composition.
    
    Args:
        configs: Dictionary of {name: hyp_dict}
        output_path: Path to save the visualization
    """
    # Calculate losses for each configuration
    results = {}
    for name, hyp in configs.items():
        results[name] = calculate_loss_composition(hyp, BATCH_SIZE)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Color scheme
    colors = {
        'detection': '#2E86AB',
        'classification': '#A23B72',
        'constraint': '#F18F01',
        'lbox': '#06A77D',
        'lobj': '#D8973C',
        'lcls_det': '#BD632F',
    }
    
    config_names = list(results.keys())
    n_configs = len(config_names)
    
    # 1. Stacked Bar Chart - Loss Components
    ax1 = plt.subplot(2, 3, 1)
    width = 0.6
    x = np.arange(n_configs)
    
    detection_vals = [results[name]['detection'] for name in config_names]
    classification_vals = [results[name]['classification'] for name in config_names]
    constraint_vals = [results[name]['constraint'] for name in config_names]
    
    ax1.bar(x, detection_vals, width, label='Detection', color=colors['detection'], alpha=0.8)
    ax1.bar(x, classification_vals, width, bottom=detection_vals, 
            label='Classification', color=colors['classification'], alpha=0.8)
    ax1.bar(x, constraint_vals, width, 
            bottom=[d+c for d, c in zip(detection_vals, classification_vals)],
            label='Constraint', color=colors['constraint'], alpha=0.8)
    
    ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Components (Absolute Values)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=15, ha='right')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add total loss values on top
    for i, name in enumerate(config_names):
        total = results[name]['total']
        ax1.text(i, total + 0.2, f'{total:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 2. Pie Charts - Loss Distribution
    for idx, name in enumerate(config_names):
        ax = plt.subplot(2, 3, idx + 2)
        
        sizes = [
            results[name]['detection'],
            results[name]['classification'],
            results[name]['constraint']
        ]
        labels = [
            f"Detection\n{results[name]['detection_pct']:.1f}%",
            f"Classification\n{results[name]['classification_pct']:.1f}%",
            f"Constraint\n{results[name]['constraint_pct']:.1f}%"
        ]
        pie_colors = [colors['detection'], colors['classification'], colors['constraint']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                           autopct='', startangle=90, 
                                           wedgeprops={'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 2})
        
        # Make percentage text bold
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        ax.set_title(f'{name}\nTotal Loss: {results[name]["total"]:.2f}', 
                    fontsize=12, fontweight='bold', pad=10)
    
    # 3. Detailed Breakdown - Detection Components
    ax5 = plt.subplot(2, 3, 5)
    
    x = np.arange(n_configs)
    width = 0.25
    
    lbox_vals = [results[name]['lbox'] for name in config_names]
    lobj_vals = [results[name]['lobj'] for name in config_names]
    lcls_det_vals = [results[name]['lcls_det'] for name in config_names]
    
    ax5.bar(x - width, lbox_vals, width, label='Box Loss', color=colors['lbox'], alpha=0.8)
    ax5.bar(x, lobj_vals, width, label='Obj Loss', color=colors['lobj'], alpha=0.8)
    ax5.bar(x + width, lcls_det_vals, width, label='Cls Loss (Det)', color=colors['lcls_det'], alpha=0.8)
    
    ax5.set_ylabel('Loss Value (Scaled)', fontsize=12, fontweight='bold')
    ax5.set_title('Detection Loss Breakdown', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(config_names, rotation=15, ha='right')
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 4. Percentage Comparison Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create table data
    table_data = [['Configuration', 'Detection %', 'Classification %', 'Constraint %', 'Total Loss']]
    for name in config_names:
        row = [
            name,
            f"{results[name]['detection_pct']:.1f}%",
            f"{results[name]['classification_pct']:.1f}%",
            f"{results[name]['constraint_pct']:.1f}%",
            f"{results[name]['total']:.2f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')
    
    # Color code percentages
    for i, name in enumerate(config_names, start=1):
        # Detection
        det_pct = results[name]['detection_pct']
        cell = table[(i, 1)]
        if det_pct > 60:
            cell.set_facecolor('#C8E6C9')  # Green - detection dominant
        elif det_pct > 40:
            cell.set_facecolor('#FFF9C4')  # Yellow - balanced
        else:
            cell.set_facecolor('#FFCDD2')  # Red - detection weak
        
        # Classification
        cls_pct = results[name]['classification_pct']
        cell = table[(i, 2)]
        if cls_pct > 50:
            cell.set_facecolor('#E1BEE7')  # Purple - classification dominant
        elif cls_pct > 30:
            cell.set_facecolor('#FFF9C4')  # Yellow - balanced
        else:
            cell.set_facecolor('#FFCDD2')  # Red - classification weak
    
    ax6.set_title('Loss Distribution Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle(f'Loss Composition Comparison (Batch Size: {BATCH_SIZE})\n' + 
                 'Loss Scaling Strategy: Detection × bs | Classification (no scale) | Constraint (no scale)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_path}")
    
    return fig


def print_loss_analysis(configs):
    """Print detailed loss analysis for each configuration."""
    print("\n" + "="*80)
    print("LOSS COMPOSITION ANALYSIS")
    print("="*80)
    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Loss Scaling Strategy: Detection x bs | Classification (no scale) | Constraint (no scale)")
    print("\n" + "-"*80)
    
    for name, hyp in configs.items():
        result = calculate_loss_composition(hyp, BATCH_SIZE)
        
        print(f"\n[{name.upper()}]")
        print("-" * 80)
        print(f"Hyperparameters:")
        print(f"  box={hyp.get('box', 0.05):.3f}, obj={hyp.get('obj', 1.0):.2f}, "
              f"cls={hyp.get('cls', 0.5):.2f}, cls_task={hyp.get('cls_task', 0.3):.2f}, "
              f"constraint_weight={hyp.get('constraint_weight', 0.3):.2f}")
        
        print(f"\nLoss Components:")
        print(f"  Detection Loss:       {result['detection']:6.2f}  ({result['detection_pct']:5.1f}%)")
        print(f"    - Box Loss:         {result['lbox']:6.2f}")
        print(f"    - Obj Loss:         {result['lobj']:6.2f}")
        print(f"    - Cls Loss (Det):   {result['lcls_det']:6.2f}")
        print(f"  Classification Loss:  {result['classification']:6.2f}  ({result['classification_pct']:5.1f}%)")
        print(f"  Constraint Loss:      {result['constraint']:6.2f}  ({result['constraint_pct']:5.1f}%)")
        print(f"  {'-'*40}")
        print(f"  Total Loss:          {result['total']:6.2f}")
        
        # Analysis
        print(f"\nAnalysis:")
        if result['detection_pct'] > 60:
            print(f"  [+] Detection-dominant configuration (good for improving mAP)")
        elif result['detection_pct'] < 35:
            print(f"  [!] Detection may be under-trained (low mAP risk)")
        else:
            print(f"  [+] Balanced detection contribution")
        
        if result['classification_pct'] > 50:
            print(f"  [+] Classification-dominant (good for view accuracy)")
        elif result['classification_pct'] < 25:
            print(f"  [!] Classification may be under-trained")
        else:
            print(f"  [+] Balanced classification contribution")
        
        if result['constraint_pct'] > 25:
            print(f"  [!] Constraint may be too dominant")
        elif result['constraint_pct'] < 5:
            print(f"  [!] Constraint may be too weak")
        else:
            print(f"  [+] Appropriate constraint influence")
        
        # Recommendation
        print(f"\nBest For:")
        if result['detection_pct'] > 60:
            print(f"  - Maximizing detection mAP")
            print(f"  - Improving bbox precision and recall")
        elif result['classification_pct'] > 50:
            print(f"  - Maximizing classification accuracy")
            print(f"  - Distinguishing similar views")
        else:
            print(f"  - Balanced training of both tasks")
            print(f"  - General-purpose configuration")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to run the visualization."""
    # Define hyperparameter files
    hyp_files = {
        'Balanced': 'yolov5c/data/hyps/hyp.balanced_v2.yaml',
        'Detection Priority': 'yolov5c/data/hyps/hyp.detection_priority_v2.yaml',
        'Classification Priority': 'yolov5c/data/hyps/hyp.classification_priority_v2.yaml',
    }
    
    # Load configurations
    configs = {}
    print("\n[*] Loading hyperparameter configurations...")
    for name, path in hyp_files.items():
        if Path(path).exists():
            configs[name] = load_hyp(path)
            print(f"  [OK] Loaded: {name} ({path})")
        else:
            print(f"  [FAIL] Not found: {name} ({path})")
    
    if not configs:
        print("\n[ERROR] No hyperparameter files found!")
        return
    
    # Print analysis
    print_loss_analysis(configs)
    
    # Create visualization
    print("[*] Creating visualization...")
    fig = create_loss_visualization(configs)
    plt.show()
    
    print("\n" + "="*80)
    print("USAGE GUIDE")
    print("="*80)
    print("\n1. Balanced Configuration (hyp.balanced_v2.yaml)")
    print("   Use when: Equal importance for detection and classification")
    print("   Command: python train.py --hyp data/hyps/hyp.balanced_v2.yaml ...")
    
    print("\n2. Detection Priority Configuration (hyp.detection_priority_v2.yaml)")
    print("   Use when: mAP is most important, classification already good")
    print("   Command: python train.py --hyp data/hyps/hyp.detection_priority_v2.yaml ...")
    
    print("\n3. Classification Priority Configuration (hyp.classification_priority_v2.yaml)")
    print("   Use when: View accuracy is critical, detection acceptable")
    print("   Command: python train.py --hyp data/hyps/hyp.classification_priority_v2.yaml ...")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()

