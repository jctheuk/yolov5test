"""
Aggregate thesis results from Excel and create comparison outputs.

This script:
1. Reads thesis_results_complete.xlsx
2. Aggregates v1-v5 per model
3. Generates:
   - results/combined_metrics.csv
   - results/combined_table.tex
   - files/1760423080004_compared@2x.jpg (with table overlay)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_thesis_results(excel_path):
    """Load thesis results from Excel file."""
    df = pd.read_excel(excel_path)
    return df

def aggregate_by_model(df):
    """Aggregate metrics by model type and architecture."""
    # Group by model_type and architecture
    grouped = df.groupby(['model_type', 'architecture']).agg({
        'mAP_0.5': 'mean',
        'mAP_0.5:0.95': 'mean',
        'cls_accuracy': 'mean',
        'cls_precision': 'mean',
        'cls_recall': 'mean',
        'cls_f1_score': 'mean'
    }).reset_index()
    
    # Convert classification metrics from decimal to percentage (if < 1)
    for col in ['cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1_score']:
        grouped[col] = grouped[col].apply(lambda x: x * 100 if pd.notna(x) and x < 1 else x)
    
    return grouped

def load_yolov5original_results(analysis_file):
    """Load yolov5original results from analysis markdown."""
    results = []
    
    # Hardcoded results from YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md
    models = {
        'YOLOv5-S': {
            'versions': [96.41, 97.60, 97.29, 99.33, 98.29],
            'avg': 97.78
        },
        'YOLOv5-M': {
            'versions': [97.39, 98.63, 97.63, 99.66, 97.95],
            'avg': 98.25
        },
        'YOLOv5-L': {
            'versions': [96.73, 96.58, 96.95, 98.66, 97.95],
            'avg': 97.37
        }
    }
    
    for model_name, data in models.items():
        results.append({
            'model_type': model_name,
            'architecture': 'classify',
            'mAP_0.5': None,  # Classification only
            'mAP_0.5:0.95': None,
            'cls_accuracy': data['avg'],
            'cls_precision': None,  # Not available in summary
            'cls_recall': None,
            'cls_f1_score': None
        })
    
    return pd.DataFrame(results)

def save_to_csv(df, output_path):
    """Save aggregated metrics to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

def save_to_latex(df, output_path):
    """Save metrics to LaTeX table format."""
    latex_lines = []
    
    # Table header
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Comparison of YOLOv5 Models (Averaged over V1-V5)}")
    latex_lines.append(r"\label{tab:model_comparison}")
    latex_lines.append(r"\begin{tabular}{l|l|cc|cccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"Model & Arch & mAP@0.5 & mAP@0.5:0.95 & Cls Acc. & Cls Prec. & Cls Recall & Cls F1 \\")
    latex_lines.append(r"\hline")
    
    # Table rows
    for _, row in df.iterrows():
        model_name = str(row['model_type']).replace('_', r'\_')
        arch = str(row['architecture']).replace('_', r'\_')
        
        # Format metrics
        map50 = f"{row['mAP_0.5']:.3f}" if pd.notna(row['mAP_0.5']) else "N/A"
        map5095 = f"{row['mAP_0.5:0.95']:.3f}" if pd.notna(row['mAP_0.5:0.95']) else "N/A"
        acc = f"{row['cls_accuracy']:.2f}\\%" if pd.notna(row['cls_accuracy']) else "N/A"
        prec = f"{row['cls_precision']:.2f}\\%" if pd.notna(row['cls_precision']) else "N/A"
        recall = f"{row['cls_recall']:.2f}\\%" if pd.notna(row['cls_recall']) else "N/A"
        f1 = f"{row['cls_f1_score']:.2f}\\%" if pd.notna(row['cls_f1_score']) else "N/A"
        
        # Row
        row_text = f"{model_name} & {arch} & {map50} & {map5095} & {acc} & {prec} & {recall} & {f1} \\\\"
        latex_lines.append(row_text)
    
    # Table footer
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved LaTeX table to {output_path}")

def create_comparison_image(df, base_image_path, output_path):
    """Create comparison image with table overlay."""
    # Load base image
    if not os.path.exists(base_image_path):
        print(f"Base image not found: {base_image_path}, skipping image generation")
        # Create a blank white image instead
        img = Image.new('RGB', (1920, 1080), color='white')
    else:
        img = Image.open(base_image_path)
    
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_header = ImageFont.truetype("arial.ttf", 18)
        font_body = ImageFont.truetype("arial.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
    
    # Draw semi-transparent white background for text
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 200))
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    
    # Draw title
    title = "YOLOv5 Model Comparison (Averaged V1-V5)"
    draw.text((50, 30), title, fill='black', font=font_title)
    
    # Draw table header
    y_offset = 80
    header = f"{'Model':<20} {'Arch':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'Cls Acc':<10} {'Cls Prec':<10} {'Cls Recall':<12} {'Cls F1':<10}"
    draw.text((50, y_offset), header, fill='black', font=font_header)
    y_offset += 30
    
    # Draw separator
    draw.line([(50, y_offset), (img.width - 50, y_offset)], fill='black', width=2)
    y_offset += 15
    
    # Draw table rows
    for _, row in df.iterrows():
        # Format data
        model_name = str(row['model_type'])[:18]
        arch = str(row['architecture'])[:10]
        map50 = f"{row['mAP_0.5']:.3f}" if pd.notna(row['mAP_0.5']) else "N/A"
        map5095 = f"{row['mAP_0.5:0.95']:.3f}" if pd.notna(row['mAP_0.5:0.95']) else "N/A"
        acc = f"{row['cls_accuracy']:.2f}%" if pd.notna(row['cls_accuracy']) else "N/A"
        prec = f"{row['cls_precision']:.2f}%" if pd.notna(row['cls_precision']) else "N/A"
        recall = f"{row['cls_recall']:.2f}%" if pd.notna(row['cls_recall']) else "N/A"
        f1 = f"{row['cls_f1_score']:.2f}%" if pd.notna(row['cls_f1_score']) else "N/A"
        
        # Format row with fixed width
        row_text = f"{model_name:<20} {arch:<12} {map50:<10} {map5095:<14} {acc:<10} {prec:<10} {recall:<12} {f1:<10}"
        draw.text((50, y_offset), row_text, fill='black', font=font_body)
        y_offset += 22
        
        # Stop if we run out of space
        if y_offset > img.height - 100:
            break
    
    # Convert back to RGB for saving as JPEG
    img = img.convert('RGB')
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)
    print(f"Saved comparison image to {output_path}")

def main():
    # Base path (current working directory)
    base_path = os.getcwd()
    
    print("Loading thesis results from Excel...")
    excel_path = os.path.join(base_path, 'thesis_results_complete.xlsx')
    df = load_thesis_results(excel_path)
    
    print(f"Loaded {len(df)} records from Excel")
    
    print("Aggregating by model and architecture...")
    aggregated = aggregate_by_model(df)
    
    print("Loading yolov5original results...")
    original_results = load_yolov5original_results('YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md')
    
    print("Combining all results...")
    combined = pd.concat([aggregated, original_results], ignore_index=True)
    
    # Sort by model type and architecture
    combined = combined.sort_values(['model_type', 'architecture'])
    
    # Save to CSV
    csv_path = os.path.join(base_path, 'results', 'combined_metrics.csv')
    save_to_csv(combined, csv_path)
    
    # Save to LaTeX
    tex_path = os.path.join(base_path, 'results', 'combined_table.tex')
    save_to_latex(combined, tex_path)
    
    # Create comparison image
    base_image = os.path.join(base_path, 'files', '1760423080004@2x.jpg')
    output_image = os.path.join(base_path, 'files', '1760423080004_compared@2x.jpg')
    create_comparison_image(combined, base_image, output_image)
    
    print("\n=== Aggregation Complete ===")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {tex_path}")
    print(f"Image: {output_image}")
    print(f"\nTotal models: {len(combined)}")
    print(f"\nPreview of results:")
    print(combined.to_string(index=False))

if __name__ == '__main__':
    main()

