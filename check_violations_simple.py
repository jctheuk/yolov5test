#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Dataset Constraint Violations Checker for V2-V5
Check anatomical constraint violations in regurgitationV2 to V5 datasets
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class AnatomicalConstraints:
    """Anatomical constraint definitions"""
    
    def __init__(self):
        # View class definitions
        self.view_names = {
            0: 'A4C',   # Apical 4-Chamber
            1: 'PSAX',  # Parasternal Short Axis  
            2: 'PLAX'   # Parasternal Long Axis
        }
        
        # Regurgitation class definitions
        self.regurg_names = {
            0: 'AR',    # Aortic Regurgitation
            1: 'MR',    # Mitral Regurgitation
            2: 'PR',    # Pulmonary Regurgitation
            3: 'TR'     # Tricuspid Regurgitation
        }
        
        # Anatomical constraint rules: allowed regurgitation types per view
        self.constraints = {
            0: [1, 3],  # A4C: only allows MR (1), TR (3)
            1: [2, 3],  # PSAX: only allows PR (2), TR (3)
            2: [0, 1],  # PLAX: only allows AR (0), MR (1)
        }
    
    def is_violation(self, view_class, detection_class):
        """Check if detection violates anatomical constraints"""
        if view_class not in self.constraints:
            return False
        allowed_classes = self.constraints[view_class]
        return detection_class not in allowed_classes
    
    def get_violation_type(self, view_class, detection_class):
        """Get violation type description"""
        if not self.is_violation(view_class, detection_class):
            return None
        view_name = self.view_names.get(view_class, f'VIEW_{view_class}')
        regurg_name = self.regurg_names.get(detection_class, f'REGURG_{detection_class}')
        return f"{view_name}_{regurg_name}"


class DatasetViolationChecker:
    """Dataset constraint violation checker"""
    
    def __init__(self):
        self.constraints = AnatomicalConstraints()
        self.results = {}
    
    def parse_label_file(self, label_path):
        """Parse label file and extract detections and view class"""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) < 2:
                return None, None
            
            detections = []
            view_class = None
            
            for line in lines:
                parts = line.split()
                
                # Check if this is classification line (3 elements, all 0 or 1)
                if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                    # Find which position has '1' to get view class
                    for i, val in enumerate(parts):
                        if val == '1':
                            view_class = i
                            break
                        
                # Check if this is detection line (5+ elements)
                elif len(parts) >= 5:
                    try:
                        detection_class = int(parts[0])
                        detections.append(detection_class)
                    except ValueError:
                        continue
            
            return detections, view_class
            
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return None, None
    
    def check_dataset(self, dataset_path, dataset_name):
        """Check entire dataset for constraint violations"""
        print(f"\n=== Checking dataset: {dataset_name} ===")
        print(f"Path: {dataset_path}")
        
        dataset_results = {
            'dataset_name': dataset_name,
            'total_files': 0,
            'parsed_files': 0,
            'violation_files': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_split': defaultdict(int),
            'violation_details': []
        }
        
        # Check each split (train, valid, test)
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            labels_dir = os.path.join(dataset_path, split, 'labels')
            
            if not os.path.exists(labels_dir):
                print(f"  Warning: {labels_dir} not found, skipping...")
                continue
            
            print(f"  Checking {split} split...")
            
            split_files = 0
            split_violations = 0
            
            # Process all label files
            for label_file in os.listdir(labels_dir):
                if not label_file.endswith('.txt'):
                    continue
                
                label_path = os.path.join(labels_dir, label_file)
                dataset_results['total_files'] += 1
                split_files += 1
                
                # Parse label file
                detections, view_class = self.parse_label_file(label_path)
                
                if detections is None or view_class is None:
                    continue
                
                dataset_results['parsed_files'] += 1
                
                # Check each detection for violations
                file_violations = []
                
                for detection_class in detections:
                    if self.constraints.is_violation(view_class, detection_class):
                        violation_type = self.constraints.get_violation_type(view_class, detection_class)
                        
                        violation_info = {
                            'file': label_file,
                            'split': split,
                            'view_class': view_class,
                            'view_name': self.constraints.view_names[view_class],
                            'detection_class': detection_class,
                            'detection_name': self.constraints.regurg_names[detection_class],
                            'violation_type': violation_type
                        }
                        
                        file_violations.append(violation_info)
                        dataset_results['violations_by_type'][violation_type] += 1
                
                # Record file if it has violations
                if file_violations:
                    dataset_results['violation_files'] += 1
                    dataset_results['violations_by_split'][split] += 1
                    split_violations += 1
                    dataset_results['violation_details'].extend(file_violations)
            
            print(f"    {split}: {split_files} files, {split_violations} violations")
        
        # Calculate statistics
        if dataset_results['parsed_files'] > 0:
            violation_rate = (dataset_results['violation_files'] / dataset_results['parsed_files']) * 100
            print(f"\n  Summary:")
            print(f"    Total files: {dataset_results['total_files']}")
            print(f"    Successfully parsed: {dataset_results['parsed_files']}")
            print(f"    Violation files: {dataset_results['violation_files']}")
            print(f"    Violation rate: {violation_rate:.2f}%")
            
            # Show violation type distribution
            if dataset_results['violations_by_type']:
                print(f"    Violation types:")
                for vtype, count in sorted(dataset_results['violations_by_type'].items()):
                    percentage = (count / len(dataset_results['violation_details'])) * 100
                    print(f"      {vtype}: {count} ({percentage:.1f}%)")
        
        self.results[dataset_name] = dataset_results
        return dataset_results
    
    def generate_violation_files_list(self, dataset_name, output_dir):
        """Generate list of violation files"""
        if dataset_name not in self.results:
            return None
        
        dataset_results = self.results[dataset_name]
        violation_files = set()
        
        for violation in dataset_results['violation_details']:
            violation_files.add(violation['file'])
        
        # Write to file
        output_file = os.path.join(output_dir, f"{dataset_name}_constraint_violation_filenames.txt")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Constraint Violation Files for {dataset_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total violations: {len(violation_files)}\n")
            f.write(f"# Violation rate: {(len(violation_files) / dataset_results['parsed_files'] * 100):.2f}%\n")
            f.write("# Format: filename\n\n")
            
            for filename in sorted(violation_files):
                f.write(f"{filename}\n")
        
        print(f"  Violation files list saved: {output_file}")
        return output_file
    
    def export_json(self, output_dir):
        """Export detailed results as JSON"""
        output_file = os.path.join(output_dir, f"constraint_violations_v2_v5_analysis.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert defaultdict to regular dict for JSON serialization
        export_data = {}
        for dataset_name, results in self.results.items():
            export_data[dataset_name] = {
                'dataset_name': results['dataset_name'],
                'total_files': results['total_files'],
                'parsed_files': results['parsed_files'],
                'violation_files': results['violation_files'],
                'violations_by_type': dict(results['violations_by_type']),
                'violations_by_split': dict(results['violations_by_split']),
                'violation_details': results['violation_details']
            }
        
        # Add summary statistics
        export_data['summary'] = self.generate_summary()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed analysis results exported: {output_file}")
        return output_file
    
    def generate_summary(self):
        """Generate cross-dataset summary statistics"""
        summary = {
            'total_datasets': len(self.results),
            'total_files_across_datasets': 0,
            'total_violations_across_datasets': 0,
            'violation_types_summary': defaultdict(int),
            'dataset_comparison': {}
        }
        
        for dataset_name, results in self.results.items():
            summary['total_files_across_datasets'] += results['parsed_files']
            summary['total_violations_across_datasets'] += results['violation_files']
            
            # Accumulate violation types
            for vtype, count in results['violations_by_type'].items():
                summary['violation_types_summary'][vtype] += count
            
            # Dataset comparison info
            if results['parsed_files'] > 0:
                violation_rate = (results['violation_files'] / results['parsed_files']) * 100
                summary['dataset_comparison'][dataset_name] = {
                    'files': results['parsed_files'],
                    'violations': results['violation_files'],
                    'violation_rate': round(violation_rate, 2)
                }
        
        # Convert defaultdict
        summary['violation_types_summary'] = dict(summary['violation_types_summary'])
        
        return summary
    
    def print_summary_report(self):
        """Print summary report"""
        print("\n" + "="*60)
        print("V2-V5 Dataset Constraint Violation Analysis Summary Report")
        print("="*60)
        
        if not self.results:
            print("No data to analyze")
            return
        
        summary = self.generate_summary()
        
        # Overview statistics
        print(f"\nOverview Statistics:")
        print(f"   Number of datasets: {summary['total_datasets']}")
        print(f"   Total files: {summary['total_files_across_datasets']}")
        print(f"   Total violations: {summary['total_violations_across_datasets']}")
        
        if summary['total_files_across_datasets'] > 0:
            overall_rate = (summary['total_violations_across_datasets'] / summary['total_files_across_datasets']) * 100
            print(f"   Overall violation rate: {overall_rate:.2f}%")
        
        # Dataset comparison
        print(f"\nDataset Comparison:")
        print("   Dataset".ljust(15) + "Files".ljust(10) + "Violations".ljust(12) + "Rate")
        print("   " + "-"*45)
        
        for dataset_name, stats in summary['dataset_comparison'].items():
            print(f"   {dataset_name.ljust(15)}{str(stats['files']).ljust(10)}{str(stats['violations']).ljust(12)}{stats['violation_rate']:.2f}%")
        
        # Violation type analysis
        print(f"\nViolation Type Distribution:")
        if summary['violation_types_summary']:
            total_violations = sum(summary['violation_types_summary'].values())
            print("   Type".ljust(12) + "Count".ljust(8) + "Percentage")
            print("   " + "-"*25)
            
            for vtype, count in sorted(summary['violation_types_summary'].items()):
                percentage = (count / total_violations) * 100 if total_violations > 0 else 0
                print(f"   {vtype.ljust(12)}{str(count).ljust(8)}{percentage:.1f}%")
        else:
            print("   No violations found!")
        
        print("\n" + "="*60)


def main():
    """Main function: Check V2-V5 datasets for constraint violations"""
    print("YOLOv5WithClassification V2-V5 Dataset Constraint Violation Checker")
    print("Based on anatomical constraint rules (ANATOMICAL_CONSTRAINTS_RULES_COMPLETE.md)")
    print("-" * 60)
    
    # Initialize checker
    checker = DatasetViolationChecker()
    
    # Define datasets to check
    datasets = {
        'regurgitationV2': './regurgitationV2',
        'regurgitationV3': './regurgitationV3', 
        'regurgitationV4': './regurgitationV4',
        'regurgitationV5': './regurgitationV5'
    }
    
    # Check each dataset
    found_datasets = 0
    
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            checker.check_dataset(dataset_path, dataset_name)
            
            # Generate violation files list
            checker.generate_violation_files_list(dataset_name, './violation_analysis')
            found_datasets += 1
        else:
            print(f"\nDataset not found: {dataset_path}")
    
    if found_datasets == 0:
        print("\nNo datasets found, please check paths")
        return
    
    # Export detailed analysis
    checker.export_json('./violation_analysis')
    
    # Print summary report
    checker.print_summary_report()
    
    print(f"\nAnalysis complete! Checked {found_datasets} datasets")
    print("Detailed results saved in ./violation_analysis/ directory")


if __name__ == "__main__":
    main()
