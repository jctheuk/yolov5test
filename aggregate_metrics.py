"""
Aggregate all per-class metrics from YOLOv5 classification models
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def aggregate_all_metrics(metrics_dir='classification_metrics'):
    """Aggregate metrics from all models"""
    metrics_dir = Path(metrics_dir)
    
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return None
    
    # Find all metrics CSV files
    csv_files = list(metrics_dir.glob('*_metrics.csv'))
    
    if not csv_files:
        print(f"No metrics files found in {metrics_dir}")
        return None
    
    print(f"Found {len(csv_files)} metrics files")
    
    # Aggregate all metrics
    all_metrics = []
    
    for csv_file in sorted(csv_files):
        model_name = csv_file.stem.replace('_metrics', '')
        df = pd.DataFrame(csv_file)
        
        # Parse model type and version
        if 'classifys' in model_name:
            model_size = 'Small'
        elif 'classifym' in model_name:
            model_size = 'Medium'
        elif 'classifyl' in model_name:
            model_size = 'Large'
        else:
            model_size = 'Unknown'
        
        version = model_name.split('_')[-1].upper()
        
        df['Model_Size'] = model_size
        df['Version'] = version
        df['Model_Name'] = model_name
        
        all_metrics.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save combined metrics
    output_file = metrics_dir / 'all_models_metrics.csv'
    combined_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n✅ Saved combined metrics: {output_file}")
    
    return combined_df


def create_summary_tables(df, output_dir='classification_metrics'):
    """Create summary tables for the report"""
    output_dir = Path(output_dir)
    
    # Per-class average across all models
    class_summary = df.groupby('Class').agg({
        'Precision': ['mean', 'std', 'min', 'max'],
        'Recall': ['mean', 'std', 'min', 'max'],
        'F1-Score': ['mean', 'std', 'min', 'max'],
        'Support': 'mean'
    }).round(4)
    
    class_summary_file = output_dir / 'class_summary.csv'
    class_summary.to_csv(class_summary_file)
    print(f"✅ Saved class summary: {class_summary_file}")
    
    # Per-model performance
    model_summary = df.groupby(['Model_Size', 'Version', 'Model_Name']).agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean'
    }).round(4)
    
    model_summary_file = output_dir / 'model_summary.csv'
    model_summary.to_csv(model_summary_file)
    print(f"✅ Saved model summary: {model_summary_file}")
    
    # Best performing models per class
    best_per_class = df.loc[df.groupby('Class')['F1-Score'].idxmax()]
    best_file = output_dir / 'best_per_class.csv'
    best_per_class[['Class', 'Model_Name', 'Precision', 'Recall', 'F1-Score']].to_csv(best_file, index=False, float_format='%.4f')
    print(f"✅ Saved best per class: {best_file}")
    
    return class_summary, model_summary, best_per_class


def create_visualizations(df, output_dir='classification_metrics'):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    
    # 1. F1-Score heatmap by model and class
    plt.figure(figsize=(14, 10))
    pivot_f1 = df.pivot_table(values='F1-Score', index=['Model_Size', 'Version'], columns='Class')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.9, vmax=1.0, cbar_kws={'label': 'F1-Score'})
    plt.title('F1-Score by Model and Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Model (Size_Version)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved F1 heatmap")
    
    # 2. Per-class performance across models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for class_name in df['Class'].unique():
            class_data = df[df['Class'] == class_name]
            ax.plot(class_data['Model_Name'], class_data[metric], marker='o', label=class_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(f'{metric} by Model and Class', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_model_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved metrics by model/class plot")
    
    # 3. Box plot of metrics by class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df.boxplot(column=metric, by='Class', ax=ax)
        ax.set_title(f'{metric} Distribution by Class')
        ax.set_xlabel('Class')
        ax.set_ylabel(metric)
        ax.set_ylim([0.85, 1.0])
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved metrics distribution plot")


def generate_markdown_report(df, output_dir='classification_metrics'):
    """Generate markdown report section"""
    output_dir = Path(output_dir)
    
    # Calculate summaries
    class_avg = df.groupby('Class').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean',
        'Support': 'mean'
    }).round(4)
    
    model_avg = df.groupby('Model_Name').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean'
    }).round(4)
    
    # Generate markdown
    md_content = """## 實際每類別指標分析

### 匯總統計

基於所有 15 個訓練模型的實際驗證結果：

#### 每類別平均性能

| 類別 | 精確率 (Precision) | 召回率 (Recall) | F1分數 | 平均樣本數 |
|------|-------------------|----------------|--------|-----------|
"""
    
    for class_name in class_avg.index:
        row = class_avg.loc[class_name]
        md_content += f"| **{class_name}** | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {int(row['Support'])} |\n"
    
    md_content += "\n#### 每模型平均性能\n\n"
    md_content += "| 模型 | 平均精確率 | 平均召回率 | 平均F1分數 |\n"
    md_content += "|------|-----------|-----------|----------|\n"
    
    for model_name in sorted(model_avg.index):
        row = model_avg.loc[model_name]
        md_content += f"| {model_name} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n"
    
    # Best performers
    best_models = model_avg.nlargest(3, 'F1-Score')
    md_content += "\n### 最佳表現模型（按F1分數）\n\n"
    md_content += "| 排名 | 模型 | 平均F1分數 |\n"
    md_content += "|------|------|----------|\n"
    
    for idx, (model_name, row) in enumerate(best_models.iterrows(), 1):
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        md_content += f"| {emoji} {idx} | {model_name} | {row['F1-Score']:.4f} |\n"
    
    # Per-class insights
    md_content += "\n### 每類別表現洞察\n\n"
    
    for class_name in class_avg.index:
        class_data = df[df['Class'] == class_name]
        avg_f1 = class_data['F1-Score'].mean()
        std_f1 = class_data['F1-Score'].std()
        min_f1 = class_data['F1-Score'].min()
        max_f1 = class_data['F1-Score'].max()
        
        md_content += f"#### {class_name}\n"
        md_content += f"- **平均F1分數**: {avg_f1:.4f} (σ={std_f1:.4f})\n"
        md_content += f"- **範圍**: {min_f1:.4f} - {max_f1:.4f}\n"
        md_content += f"- **變異性**: {'低' if std_f1 < 0.02 else '中' if std_f1 < 0.05 else '高'}\n\n"
    
    # Save markdown
    md_file = output_dir / 'metrics_report_section.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✅ Saved markdown report: {md_file}")
    
    return md_content


def main():
    print("Aggregating classification metrics...\n")
    print("="*80)
    
    # Aggregate all metrics
    df = aggregate_all_metrics()
    
    if df is None or len(df) == 0:
        print("\n⚠️ No metrics data found. Please run compute_all_metrics.ps1 first.")
        return
    
    print(f"\n📊 Total records: {len(df)}")
    print(f"📦 Models: {df['Model_Name'].nunique()}")
    print(f"📋 Classes: {df['Class'].nunique()}")
    
    # Create summary tables
    print("\n" + "="*80)
    print("Creating summary tables...")
    print("="*80 + "\n")
    create_summary_tables(df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80 + "\n")
    create_visualizations(df)
    
    # Generate markdown report
    print("\n" + "="*80)
    print("Generating markdown report...")
    print("="*80 + "\n")
    md_content = generate_markdown_report(df)
    
    print("\n" + "="*80)
    print("✅ All aggregation complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - classification_metrics/all_models_metrics.csv")
    print("  - classification_metrics/class_summary.csv")
    print("  - classification_metrics/model_summary.csv")
    print("  - classification_metrics/best_per_class.csv")
    print("  - classification_metrics/f1_heatmap.png")
    print("  - classification_metrics/metrics_by_model_class.png")
    print("  - classification_metrics/metrics_distribution_by_class.png")
    print("  - classification_metrics/metrics_report_section.md")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()



