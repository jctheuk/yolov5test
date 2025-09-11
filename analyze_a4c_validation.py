#!/usr/bin/env python3
"""
Analyze A4C class distribution in validation set
"""

import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

def analyze_validation_labels():
    """Analyze classification labels in validation set"""
    
    # Path to validation labels
    labels_dir = "Regurgitation-YOLODataset-Detection/valid/labels"
    
    # Count classification labels
    classification_counts = Counter()
    total_files = 0
    a4c_files = []
    psax_files = []
    plax_files = []
    
    # Process all label files
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        total_files += 1
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Find classification line (usually line 2)
            classification_line = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and len(line.split()) == 3:
                    # This looks like a classification line (3 numbers)
                    classification_line = line
                    break
            
            if classification_line:
                parts = classification_line.split()
                if len(parts) == 3:
                    # Parse one-hot encoding: [A4C, PSAX, PLAX]
                    a4c, psax, plax = map(int, parts)
                    
                    if a4c == 1:
                        classification_counts['A4C'] += 1
                        a4c_files.append(os.path.basename(label_file))
                    elif psax == 1:
                        classification_counts['PSAX'] += 1
                        psax_files.append(os.path.basename(label_file))
                    elif plax == 1:
                        classification_counts['PLAX'] += 1
                        plax_files.append(os.path.basename(label_file))
                    else:
                        print(f"Warning: No valid classification found in {label_file}")
                        print(f"  Classification line: {classification_line}")
            else:
                print(f"Warning: No classification line found in {label_file}")
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Print results
    print("=" * 60)
    print("VALIDATION SET CLASSIFICATION ANALYSIS")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print()
    
    print("Classification Distribution:")
    for class_name, count in classification_counts.items():
        percentage = (count / total_files) * 100
        print(f"  {class_name}: {count} files ({percentage:.1f}%)")
    
    print()
    print("A4C Files (first 10):")
    for i, filename in enumerate(a4c_files[:10]):
        print(f"  {i+1}. {filename}")
    
    if len(a4c_files) > 10:
        print(f"  ... and {len(a4c_files) - 10} more")
    
    print()
    print("PSAX Files (first 10):")
    for i, filename in enumerate(psax_files[:10]):
        print(f"  {i+1}. {filename}")
    
    if len(psax_files) > 10:
        print(f"  ... and {len(psax_files) - 10} more")
    
    print()
    print("PLAX Files (first 10):")
    for i, filename in enumerate(plax_files[:10]):
        print(f"  {i+1}. {filename}")
    
    if len(plax_files) > 10:
        print(f"  ... and {len(plax_files) - 10} more")
    
    # Create visualization
    if classification_counts:
        plt.figure(figsize=(10, 6))
        
        # Bar chart
        plt.subplot(1, 2, 1)
        classes = list(classification_counts.keys())
        counts = list(classification_counts.values())
        colors = ['red', 'green', 'blue']
        
        bars = plt.bar(classes, counts, color=colors, alpha=0.7)
        plt.title('Classification Distribution in Validation Set')
        plt.xlabel('Class')
        plt.ylabel('Number of Files')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Classification Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig('validation_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as 'validation_classification_analysis.png'")
    
    return classification_counts, a4c_files, psax_files, plax_files

if __name__ == "__main__":
    analyze_validation_labels()
