"""
Constraint Violation Detection Script
Identifies data that violates anatomical constraints in echocardiogram dataset
"""

import os
import glob
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class ConstraintViolationDetector:
    """
    Detects anatomical constraint violations in echocardiogram dataset.
    
    Anatomical Constraints:
    - A4C View: Only MR, TR allowed (AR, PR forbidden)
    - PSAX View: Only PR, TR allowed (AR, MR forbidden)  
    - PLAX View: Only AR, MR allowed (PR, TR forbidden)
    """
    
    def __init__(self, dataset_path: str = "regurgitationV1"):
        self.dataset_path = dataset_path
        
        # Define anatomical constraints
        self.constraints = {
            0: [1, 3],  # A4C: MR, TR (Mitral, Tricuspid)
            1: [2, 3],  # PSAX: PR, TR (Pulmonary, Tricuspid)
            2: [0, 1],  # PLAX: AR, MR (Aortic, Mitral)
        }
        
        # View and detection names
        self.view_names = ['A4C', 'PSAX', 'PLAX']
        self.detection_names = ['AR', 'MR', 'PR', 'TR']
        
        # Violation tracking
        self.violations = []
        self.violation_stats = defaultdict(int)
        self.file_violations = defaultdict(list)
        
    def parse_label_file(self, label_file: str) -> Tuple[List[int], int, str]:
        """
        Parse a label file to extract detection classes and view class.
        
        Args:
            label_file: Path to label file
            
        Returns:
            Tuple of (detection_classes, view_class, filename)
        """
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            detection_classes = []
            view_class = None
            
            # Parse detection labels (first line)
            if lines:
                detection_line = lines[0].strip()
                if detection_line:
                    parts = detection_line.split()
                    for part in parts:
                        try:
                            detection_classes.append(int(part))
                        except ValueError:
                            continue
            
            # Parse classification label (second line)
            if len(lines) >= 2:
                classification_line = lines[1].strip()
                if classification_line:
                    parts = classification_line.split()
                    if len(parts) == 3:  # One-hot encoded
                        view_class = parts.index('1') if '1' in parts else 0
                    else:
                        try:
                            view_class = int(parts[0])
                        except (ValueError, IndexError):
                            view_class = 0
            else:
                view_class = 0  # Default to A4C if no classification label
            
            return detection_classes, view_class, os.path.basename(label_file)
            
        except Exception as e:
            print(f"Error parsing {label_file}: {e}")
            return [], 0, os.path.basename(label_file)
    
    def check_constraint_violations(self, detection_classes: List[int], view_class: int, filename: str) -> List[Dict]:
        """
        Check if detection classes violate anatomical constraints for the given view.
        
        Args:
            detection_classes: List of detection class indices
            view_class: View class index (0=A4C, 1=PSAX, 2=PLAX)
            filename: Name of the file
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        allowed_classes = self.constraints.get(view_class, [])
        view_name = self.view_names[view_class]
        
        for det_class in detection_classes:
            if det_class not in allowed_classes:
                violation = {
                    'filename': filename,
                    'view_class': view_class,
                    'view_name': view_name,
                    'detection_class': det_class,
                    'detection_name': self.detection_names[det_class],
                    'allowed_classes': [self.detection_names[cls] for cls in allowed_classes],
                    'violation_type': f"{self.detection_names[det_class]} detected in {view_name} view"
                }
                violations.append(violation)
        
        return violations
    
    def scan_dataset(self) -> Dict:
        """
        Scan the entire dataset for constraint violations.
        
        Returns:
            Dictionary containing violation statistics
        """
        print("Scanning dataset for anatomical constraint violations...")
        print("=" * 60)
        
        total_files = 0
        violation_files = 0
        total_violations = 0
        
        # Scan all splits
        for split in ['train', 'valid', 'test']:
            labels_path = os.path.join(self.dataset_path, split, 'labels')
            if not os.path.exists(labels_path):
                print(f"Warning: {labels_path} not found, skipping...")
                continue
            
            print(f"\nScanning {split} split...")
            label_files = glob.glob(os.path.join(labels_path, '*.txt'))
            
            for label_file in label_files:
                total_files += 1
                
                # Parse the label file
                detection_classes, view_class, filename = self.parse_label_file(label_file)
                
                # Check for violations
                violations = self.check_constraint_violations(detection_classes, view_class, filename)
                
                if violations:
                    violation_files += 1
                    total_violations += len(violations)
                    
                    # Store violations
                    self.violations.extend(violations)
                    self.file_violations[filename] = violations
                    
                    # Update statistics
                    for violation in violations:
                        key = f"{violation['view_name']}_{violation['detection_name']}"
                        self.violation_stats[key] += 1
        
        # Generate summary
        summary = {
            'total_files': total_files,
            'violation_files': violation_files,
            'total_violations': total_violations,
            'violation_rate': (violation_files / total_files * 100) if total_files > 0 else 0,
            'violations_per_file': total_violations / violation_files if violation_files > 0 else 0
        }
        
        return summary
    
    def print_violation_summary(self, summary: Dict):
        """Print a summary of constraint violations."""
        print("\nCONSTRAINT VIOLATION SUMMARY")
        print("=" * 60)
        print(f"Total files scanned: {summary['total_files']}")
        print(f"Files with violations: {summary['violation_files']}")
        print(f"Total violations: {summary['total_violations']}")
        print(f"Violation rate: {summary['violation_rate']:.2f}%")
        print(f"Average violations per file: {summary['violations_per_file']:.2f}")
        
        print("\nVIOLATION BREAKDOWN BY TYPE:")
        print("-" * 40)
        for violation_type, count in sorted(self.violation_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{violation_type}: {count} violations")
    
    def print_violating_files(self, max_files: int = 20):
        """Print names of files that violate constraints."""
        print(f"\nFILES WITH CONSTRAINT VIOLATIONS (showing first {max_files}):")
        print("-" * 60)
        
        count = 0
        for filename, violations in self.file_violations.items():
            if count >= max_files:
                print(f"... and {len(self.file_violations) - max_files} more files")
                break
                
            print(f"\nFile: {filename}")
            for violation in violations:
                print(f"  - {violation['violation_type']}")
            count += 1
    
    def create_violation_report(self, output_file: str = "constraint_violations_report.txt"):
        """Create a detailed violation report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ANATOMICAL CONSTRAINT VIOLATIONS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ANATOMICAL CONSTRAINTS:\n")
            f.write("-" * 30 + "\n")
            for view_idx, allowed_classes in self.constraints.items():
                view_name = self.view_names[view_idx]
                allowed_names = [self.detection_names[cls] for cls in allowed_classes]
                f.write(f"{view_name}: {', '.join(allowed_names)}\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total files: {len(self.file_violations)}\n")
            f.write(f"Total violations: {len(self.violations)}\n")
            
            f.write(f"\nVIOLATION BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for violation_type, count in sorted(self.violation_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{violation_type}: {count}\n")
            
            f.write(f"\nDETAILED VIOLATIONS BY FILE:\n")
            f.write("-" * 40 + "\n")
            for filename, violations in self.file_violations.items():
                f.write(f"\nFile: {filename}\n")
                for violation in violations:
                    f.write(f"  - {violation['violation_type']}\n")
                    f.write(f"    Allowed in {violation['view_name']}: {', '.join(violation['allowed_classes'])}\n")
        
        print(f"Detailed report saved to: {output_file}")
    
    def create_violation_visualization(self):
        """Create visualization of constraint violations."""
        if not self.violations:
            print("No violations found to visualize.")
            return
        
        # Prepare data for visualization
        violation_data = []
        for violation in self.violations:
            violation_data.append({
                'View': violation['view_name'],
                'Detection': violation['detection_name'],
                'Count': 1
            })
        
        df = pd.DataFrame(violation_data)
        
        # Create violation heatmap
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Violation counts by view and detection
        plt.subplot(2, 2, 1)
        pivot_df = df.groupby(['View', 'Detection']).size().unstack(fill_value=0)
        sns.heatmap(pivot_df, annot=True, fmt='d', cmap='Reds')
        plt.title('Constraint Violations by View and Detection')
        plt.xlabel('Detection Class')
        plt.ylabel('View Type')
        
        # Subplot 2: Violation counts by detection class
        plt.subplot(2, 2, 2)
        detection_counts = df['Detection'].value_counts()
        detection_counts.plot(kind='bar', color='red', alpha=0.7)
        plt.title('Violations by Detection Class')
        plt.xlabel('Detection Class')
        plt.ylabel('Number of Violations')
        plt.xticks(rotation=45)
        
        # Subplot 3: Violation counts by view
        plt.subplot(2, 2, 3)
        view_counts = df['View'].value_counts()
        view_counts.plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Violations by View Type')
        plt.xlabel('View Type')
        plt.ylabel('Number of Violations')
        plt.xticks(rotation=45)
        
        # Subplot 4: Violation rate by view
        plt.subplot(2, 2, 4)
        total_by_view = df['View'].value_counts()
        # Calculate violation rate (this is a simplified calculation)
        violation_rates = total_by_view / total_by_view.sum() * 100
        violation_rates.plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Violation Distribution by View')
        plt.xlabel('View Type')
        plt.ylabel('Percentage of Total Violations (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('constraint_violations_analysis.png', dpi=300, bbox_inches='tight')
        print("Violation analysis chart saved as 'constraint_violations_analysis.png'")
        plt.show()
    
    def get_violating_file_names(self) -> List[str]:
        """Get a list of all files that violate constraints."""
        return list(self.file_violations.keys())
    
    def get_violations_by_file(self, filename: str) -> List[Dict]:
        """Get violations for a specific file."""
        return self.file_violations.get(filename, [])
    
    def run_analysis(self):
        """Run complete constraint violation analysis."""
        print("ANATOMICAL CONSTRAINT VIOLATION DETECTOR")
        print("=" * 60)
        print("Detecting violations of echocardiogram anatomical constraints...")
        
        # Scan dataset
        summary = self.scan_dataset()
        
        # Print summary
        self.print_violation_summary(summary)
        
        # Print violating files
        self.print_violating_files()
        
        # Create detailed report
        self.create_violation_report()
        
        # Create visualization
        if self.violations:
            self.create_violation_visualization()
        
        return {
            'summary': summary,
            'violations': self.violations,
            'violating_files': self.get_violating_file_names(),
            'violation_stats': dict(self.violation_stats)
        }


def main():
    """Main function to run constraint violation detection."""
    print("Starting Anatomical Constraint Violation Detection...")
    
    # Initialize detector
    detector = ConstraintViolationDetector(dataset_path="regurgitationV1")
    
    # Run analysis
    results = detector.run_analysis()
    
    print("\nAnalysis completed!")
    print(f"Found {len(results['violating_files'])} files with constraint violations")
    
    # Print some example violating files
    if results['violating_files']:
        print(f"\nExample violating files:")
        for i, filename in enumerate(results['violating_files'][:5]):
            violations = detector.get_violations_by_file(filename)
            print(f"{i+1}. {filename}: {len(violations)} violations")
            for violation in violations[:2]:  # Show first 2 violations
                print(f"   - {violation['violation_type']}")
    
    return results


if __name__ == "__main__":
    results = main()

