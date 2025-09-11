#!/usr/bin/env python3
"""
Simple analysis of validation labels
"""

import os
import glob

def analyze_labels():
    labels_dir = "Regurgitation-YOLODataset-Detection/valid/labels"
    
    a4c_count = 0
    psax_count = 0
    plax_count = 0
    total = 0
    a4c_files = []
    psax_files = []
    plax_files = []
    
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Look for classification line (line 2, format: "0 1 0" or "1 0 0" etc.)
            if len(lines) >= 2:
                classification_line = lines[1].strip()
                parts = classification_line.split()
                
                if len(parts) == 3:
                    a4c, psax, plax = map(int, parts)
                    total += 1
                    
                    if a4c == 1:
                        a4c_count += 1
                        a4c_files.append(os.path.basename(label_file))
                    elif psax == 1:
                        psax_count += 1
                        psax_files.append(os.path.basename(label_file))
                    elif plax == 1:
                        plax_count += 1
                        plax_files.append(os.path.basename(label_file))
                        
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    print("=" * 60)
    print("VALIDATION SET CLASSIFICATION ANALYSIS")
    print("=" * 60)
    print(f"Total files: {total}")
    print(f"A4C: {a4c_count} ({a4c_count/total*100:.1f}%)")
    print(f"PSAX: {psax_count} ({psax_count/total*100:.1f}%)")
    print(f"PLAX: {plax_count} ({plax_count/total*100:.1f}%)")
    
    print("\nA4C Files (first 10):")
    for i, filename in enumerate(a4c_files[:10]):
        print(f"  {i+1}. {filename}")
    
    print("\nPSAX Files (first 10):")
    for i, filename in enumerate(psax_files[:10]):
        print(f"  {i+1}. {filename}")
    
    print("\nPLAX Files (first 10):")
    for i, filename in enumerate(plax_files[:10]):
        print(f"  {i+1}. {filename}")
    
    return a4c_count, psax_count, plax_count, total

if __name__ == "__main__":
    analyze_labels()
