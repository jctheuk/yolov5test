#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Constraint Violations Checker for V2-V5
檢查 regurgitationV2 到 V5 資料集的解剖學約束違反
Based on anatomical constraints from ANATOMICAL_CONSTRAINTS_RULES_COMPLETE.md
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime


class AnatomicalConstraints:
    """
    解剖學約束定義
    Based on yolov5c/utils/anatomical_constraints.py
    """
    
    def __init__(self):
        # 視圖類別定義
        self.view_names = {
            0: 'A4C',   # Apical 4-Chamber
            1: 'PSAX',  # Parasternal Short Axis  
            2: 'PLAX'   # Parasternal Long Axis
        }
        
        # 反流類別定義
        self.regurg_names = {
            0: 'AR',    # Aortic Regurgitation
            1: 'MR',    # Mitral Regurgitation
            2: 'PR',    # Pulmonary Regurgitation
            3: 'TR'     # Tricuspid Regurgitation
        }
        
        # 解剖學約束規則：每個視圖允許的反流類型
        self.constraints = {
            0: [1, 3],  # A4C: 只允許 MR (1), TR (3)
            1: [2, 3],  # PSAX: 只允許 PR (2), TR (3)
            2: [0, 1],  # PLAX: 只允許 AR (0), MR (1)
        }
        
        # 軟權重定義（用於分析違反嚴重程度）
        self.soft_weights = {
            0: {1: 1.0, 3: 1.0, 0: 0.1, 2: 0.1},  # A4C
            1: {2: 1.0, 3: 1.0, 0: 0.1, 1: 0.1},  # PSAX
            2: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.1},  # PLAX (PR impossible)
        }
    
    def is_violation(self, view_class, detection_class):
        """
        檢查是否違反解剖約束
        
        Args:
            view_class: 視圖類別 (0=A4C, 1=PSAX, 2=PLAX)
            detection_class: 檢測到的反流類別 (0=AR, 1=MR, 2=PR, 3=TR)
            
        Returns:
            bool: True 如果違反約束
        """
        if view_class not in self.constraints:
            return False
            
        allowed_classes = self.constraints[view_class]
        return detection_class not in allowed_classes
    
    def get_violation_type(self, view_class, detection_class):
        """
        獲取違反類型的描述
        
        Returns:
            str: 違反類型，例如 "A4C_AR", "PSAX_MR"
        """
        if not self.is_violation(view_class, detection_class):
            return None
            
        view_name = self.view_names.get(view_class, f'VIEW_{view_class}')
        regurg_name = self.regurg_names.get(detection_class, f'REGURG_{detection_class}')
        
        return f"{view_name}_{regurg_name}"
    
    def get_severity(self, view_class, detection_class):
        """
        獲取違反的嚴重程度
        
        Returns:
            float: 權重值，0.0 = 最嚴重，1.0 = 完全允許
        """
        if view_class not in self.soft_weights:
            return 0.5
            
        return self.soft_weights[view_class].get(detection_class, 0.0)


class DatasetViolationChecker:
    """
    資料集約束違反檢查器
    """
    
    def __init__(self):
        self.constraints = AnatomicalConstraints()
        self.results = {}
    
    def parse_label_file(self, label_path):
        """
        解析標籤文件
        
        Expected format:
        Line 1: detection_class x_center y_center width height  
        Line 2: (empty or more detections)
        Line N: view_class has_regurg regurg_present (classification one-hot encoding)
        
        Returns:
            tuple: (detections, view_class) or (None, None) if parsing fails
        """
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) < 2:
                return None, None
            
            detections = []
            view_class = None
            
            # Parse lines
            for line in lines:
                parts = line.split()
                
                # Check if this is classification line (3 elements, all 0 or 1)
                if len(parts) == 3 and all(p in ['0', '1'] for p in parts):
                    # This is the classification one-hot encoding
                    # Find which position has '1' to get view class
                    for i, val in enumerate(parts):
                        if val == '1':
                            view_class = i
                            break
                    if view_class is None:
                        # No view detected (all zeros), might be valid case
                        continue
                        
                # Check if this is detection line (5+ elements)
                elif len(parts) >= 5:
                    try:
                        detection_class = int(parts[0])
                        # x, y, w, h = float values (we don't need them for constraint checking)
                        detections.append(detection_class)
                    except ValueError:
                        continue
            
            return detections, view_class
            
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return None, None
    
    def check_dataset(self, dataset_path, dataset_name):
        """
        檢查整個資料集的約束違反
        
        Args:
            dataset_path: 資料集根目錄路徑
            dataset_name: 資料集名稱 (e.g., "regurgitationV2")
        """
        print(f"\n=== 檢查資料集: {dataset_name} ===")
        print(f"路徑: {dataset_path}")
        
        dataset_results = {
            'dataset_name': dataset_name,
            'total_files': 0,
            'parsed_files': 0,
            'violation_files': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_split': defaultdict(int),
            'violation_details': []
        }
        
        # 檢查每個分割 (train, valid, test)
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            labels_dir = os.path.join(dataset_path, split, 'labels')
            
            if not os.path.exists(labels_dir):
                print(f"  警告: {labels_dir} 不存在，跳過...")
                continue
            
            print(f"  檢查 {split} 分割...")
            
            split_files = 0
            split_violations = 0
            
            # 遍歷所有標籤文件
            for label_file in os.listdir(labels_dir):
                if not label_file.endswith('.txt'):
                    continue
                
                label_path = os.path.join(labels_dir, label_file)
                dataset_results['total_files'] += 1
                split_files += 1
                
                # 解析標籤文件
                detections, view_class = self.parse_label_file(label_path)
                
                if detections is None or view_class is None:
                    continue
                
                dataset_results['parsed_files'] += 1
                
                # 檢查每個檢測是否違反約束
                file_violations = []
                
                for detection_class in detections:
                    if self.constraints.is_violation(view_class, detection_class):
                        violation_type = self.constraints.get_violation_type(view_class, detection_class)
                        severity = self.constraints.get_severity(view_class, detection_class)
                        
                        violation_info = {
                            'file': label_file,
                            'split': split,
                            'view_class': view_class,
                            'view_name': self.constraints.view_names[view_class],
                            'detection_class': detection_class,
                            'detection_name': self.constraints.regurg_names[detection_class],
                            'violation_type': violation_type,
                            'severity': severity
                        }
                        
                        file_violations.append(violation_info)
                        dataset_results['violations_by_type'][violation_type] += 1
                
                # 如果文件有違反，記錄
                if file_violations:
                    dataset_results['violation_files'] += 1
                    dataset_results['violations_by_split'][split] += 1
                    split_violations += 1
                    dataset_results['violation_details'].extend(file_violations)
            
            print(f"    {split}: {split_files} 檔案，{split_violations} 個違反")
        
        # 計算統計資訊
        if dataset_results['parsed_files'] > 0:
            violation_rate = (dataset_results['violation_files'] / dataset_results['parsed_files']) * 100
            print(f"\n  總結:")
            print(f"    總檔案: {dataset_results['total_files']}")
            print(f"    成功解析: {dataset_results['parsed_files']}")
            print(f"    違反檔案: {dataset_results['violation_files']}")
            print(f"    違反率: {violation_rate:.2f}%")
            
            # 顯示違反類型分佈
            if dataset_results['violations_by_type']:
                print(f"    違反類型:")
                for vtype, count in sorted(dataset_results['violations_by_type'].items()):
                    percentage = (count / len(dataset_results['violation_details'])) * 100
                    print(f"      {vtype}: {count} ({percentage:.1f}%)")
        
        self.results[dataset_name] = dataset_results
        return dataset_results
    
    def generate_violation_files_list(self, dataset_name, output_dir):
        """
        生成違反檔案列表 (類似原始的 constraint_violation_filenames.txt)
        """
        if dataset_name not in self.results:
            return None
        
        dataset_results = self.results[dataset_name]
        violation_files = set()
        
        for violation in dataset_results['violation_details']:
            violation_files.add(violation['file'])
        
        # 寫入文件
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
        
        print(f"  違反檔案列表已保存: {output_file}")
        return output_file
    
    def export_json(self, output_dir):
        """
        匯出詳細結果為 JSON 格式
        """
        output_file = os.path.join(output_dir, f"constraint_violations_v2_v5_analysis.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # 轉換 defaultdict 為普通 dict 以便 JSON 序列化
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
        
        # 添加匯總統計
        export_data['summary'] = self.generate_summary()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n詳細分析結果已匯出: {output_file}")
        return output_file
    
    def generate_summary(self):
        """
        生成跨資料集的匯總統計
        """
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
            
            # 累積違反類型
            for vtype, count in results['violations_by_type'].items():
                summary['violation_types_summary'][vtype] += count
            
            # 資料集比較資訊
            if results['parsed_files'] > 0:
                violation_rate = (results['violation_files'] / results['parsed_files']) * 100
                summary['dataset_comparison'][dataset_name] = {
                    'files': results['parsed_files'],
                    'violations': results['violation_files'],
                    'violation_rate': round(violation_rate, 2)
                }
        
        # 轉換 defaultdict
        summary['violation_types_summary'] = dict(summary['violation_types_summary'])
        
        return summary
    
    def print_summary_report(self):
        """
        列印匯總報告
        """
        print("\n" + "="*60)
        print("[SUMMARY] V2-V5 資料集約束違反分析匯總報告")
        print("="*60)
        
        if not self.results:
            print("[ERROR] 沒有資料需要分析")
            return
        
        summary = self.generate_summary()
        
        # 總覽統計
        print(f"\n[STATS] 總覽統計:")
        print(f"   資料集數量: {summary['total_datasets']}")
        print(f"   總檔案數: {summary['total_files_across_datasets']}")
        print(f"   總違反數: {summary['total_violations_across_datasets']}")
        
        if summary['total_files_across_datasets'] > 0:
            overall_rate = (summary['total_violations_across_datasets'] / summary['total_files_across_datasets']) * 100
            print(f"   整體違反率: {overall_rate:.2f}%")
        
        # 各資料集比較
        print(f"\n[COMPARE] 各資料集比較:")
        print("   資料集".ljust(15) + "檔案數".ljust(10) + "違反數".ljust(10) + "違反率")
        print("   " + "-"*45)
        
        for dataset_name, stats in summary['dataset_comparison'].items():
            print(f"   {dataset_name.ljust(15)}{str(stats['files']).ljust(10)}{str(stats['violations']).ljust(10)}{stats['violation_rate']:.2f}%")
        
        # 違反類型分析
        print(f"\n[VIOLATIONS] 違反類型分佈:")
        if summary['violation_types_summary']:
            total_violations = sum(summary['violation_types_summary'].values())
            print("   類型".ljust(12) + "次數".ljust(8) + "比例")
            print("   " + "-"*25)
            
            for vtype, count in sorted(summary['violation_types_summary'].items()):
                percentage = (count / total_violations) * 100 if total_violations > 0 else 0
                print(f"   {vtype.ljust(12)}{str(count).ljust(8)}{percentage:.1f}%")
        else:
            print("   [SUCCESS] 沒有發現違反!")
        
        print("\n" + "="*60)


def main():
    """
    主函數：檢查 V2-V5 資料集的約束違反
    """
    print("[CHECK] YOLOv5WithClassification V2-V5 資料集約束違反檢查")
    print("基於解剖學約束規則 (ANATOMICAL_CONSTRAINTS_RULES_COMPLETE.md)")
    print("-" * 60)
    
    # 初始化檢查器
    checker = DatasetViolationChecker()
    
    # 定義要檢查的資料集
    datasets = {
        'regurgitationV2': './regurgitationV2',
        'regurgitationV3': './regurgitationV3', 
        'regurgitationV4': './regurgitationV4',
        'regurgitationV5': './regurgitationV5'
    }
    
    # 檢查每個資料集
    found_datasets = 0
    
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            checker.check_dataset(dataset_path, dataset_name)
            
            # 生成違反檔案列表
            checker.generate_violation_files_list(dataset_name, './violation_analysis')
            found_datasets += 1
        else:
            print(f"\n[ERROR] 資料集不存在: {dataset_path}")
    
    if found_datasets == 0:
        print("\n[ERROR] 沒有找到任何資料集，請檢查路徑")
        return
    
    # 匯出詳細分析
    checker.export_json('./violation_analysis')
    
    # 列印匯總報告
    checker.print_summary_report()
    
    print(f"\n[SUCCESS] 分析完成! 共檢查了 {found_datasets} 個資料集")
    print("[INFO] 詳細結果保存在 ./violation_analysis/ 目錄")


if __name__ == "__main__":
    main()
