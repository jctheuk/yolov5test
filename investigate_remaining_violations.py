#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調查 V2-V5 中剩餘違規的原因
檢查這些違規檔案在 V1 中的狀況
"""

import os
import json
from pathlib import Path
from collections import defaultdict


class AnatomicalConstraints:
    """解剖學約束定義"""
    
    def __init__(self):
        self.view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
        self.regurg_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
        self.constraints = {
            0: [1, 3],  # A4C: 只允許 MR (1), TR (3)
            1: [2, 3],  # PSAX: 只允許 PR (2), TR (3)  
            2: [0, 1],  # PLAX: 只允許 AR (0), MR (1)
        }
    
    def is_violation(self, view_class, detection_class):
        """檢查是否為約束違規"""
        if view_class not in self.constraints:
            return False
        allowed_classes = self.constraints[view_class]
        return detection_class not in allowed_classes
    
    def get_violation_type(self, view_class, detection_class):
        """獲取違規類型"""
        if not self.is_violation(view_class, detection_class):
            return None
        view_name = self.view_names.get(view_class, f'VIEW_{view_class}')
        regurg_name = self.regurg_names.get(detection_class, f'REGURG_{detection_class}')
        return f"{view_name}_{regurg_name}"


class ViolationInvestigator:
    """違規調查器"""
    
    def __init__(self):
        self.constraints = AnatomicalConstraints()
        self.datasets = {
            'V1': './regurgitationV1',
            'V2': './regurgitationV2', 
            'V3': './regurgitationV3',
            'V4': './regurgitationV4',
            'V5': './regurgitationV5'
        }
        
        # 儲存所有檔案位置信息
        self.file_locations = {}  # {filename: {dataset: split}}
        self.violation_files = {}  # {dataset: [violation_info]}
    
    def parse_label_file(self, file_path):
        """解析標籤檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) != 2:
                return None, None
            
            # 解析檢測資料
            detection_parts = lines[0].split()
            if len(detection_parts) != 5:
                return None, None
            
            detection_class = int(detection_parts[0])
            
            # 解析分類資料
            classification_parts = lines[1].split()
            if len(classification_parts) != 3:
                return None, None
                
            classification_data = [int(p) for p in classification_parts]
            view_class = classification_data.index(1) if 1 in classification_data else -1
            
            return detection_class, view_class
            
        except Exception as e:
            return None, None
    
    def scan_all_files(self):
        """掃描所有檔案位置"""
        print("掃描所有資料集的檔案位置...")
        
        for dataset_name, dataset_path in self.datasets.items():
            if not os.path.exists(dataset_path):
                continue
            
            for split in ['train', 'valid', 'test']:
                labels_dir = os.path.join(dataset_path, split, 'labels')
                if not os.path.exists(labels_dir):
                    continue
                
                for filename in os.listdir(labels_dir):
                    if not filename.endswith('.txt'):
                        continue
                    
                    if filename not in self.file_locations:
                        self.file_locations[filename] = {}
                    
                    self.file_locations[filename][dataset_name] = split
        
        print(f"找到 {len(self.file_locations)} 個不同的檔案")
    
    def find_violations_in_dataset(self, dataset_name):
        """找出資料集中的違規檔案"""
        dataset_path = self.datasets[dataset_name]
        violations = []
        
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(labels_dir):
                continue
            
            for filename in os.listdir(labels_dir):
                if not filename.endswith('.txt'):
                    continue
                
                file_path = os.path.join(labels_dir, filename)
                detection_class, view_class = self.parse_label_file(file_path)
                
                if detection_class is None or view_class == -1:
                    continue
                
                if self.constraints.is_violation(view_class, detection_class):
                    violation_type = self.constraints.get_violation_type(view_class, detection_class)
                    violations.append({
                        'filename': filename,
                        'split': split,
                        'detection_class': detection_class,
                        'view_class': view_class,
                        'violation_type': violation_type,
                        'file_path': file_path
                    })
        
        self.violation_files[dataset_name] = violations
        return violations
    
    def investigate_violations(self):
        """調查所有違規情況"""
        print("\n調查各資料集的違規情況...")
        
        all_investigations = {}
        
        for dataset_name in ['V2', 'V3', 'V4', 'V5']:
            print(f"\n=== 調查 {dataset_name} ===")
            violations = self.find_violations_in_dataset(dataset_name)
            
            print(f"發現 {len(violations)} 個違規")
            
            investigations = []
            
            for violation in violations:
                filename = violation['filename']
                
                # 檢查這個檔案在 V1 中的狀況
                v1_info = self.get_v1_file_info(filename)
                
                investigation = {
                    'filename': filename,
                    'current_dataset': dataset_name,
                    'current_split': violation['split'],
                    'violation_type': violation['violation_type'],
                    'v1_status': v1_info
                }
                
                investigations.append(investigation)
                
                # 顯示調查結果
                if v1_info['exists']:
                    if v1_info['split'] == violation['split']:
                        print(f"  {filename}: 在 V1 同一分割 ({v1_info['split']}) 中存在但無違規")
                    else:
                        print(f"  {filename}: 在 V1 不同分割 ({v1_info['split']} vs {violation['split']}) 中存在")
                else:
                    print(f"  {filename}: 在 V1 中不存在！")
            
            all_investigations[dataset_name] = investigations
        
        return all_investigations
    
    def get_v1_file_info(self, filename):
        """獲取檔案在 V1 中的資訊"""
        v1_path = self.datasets['V1']
        
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(v1_path, split, 'labels')
            file_path = os.path.join(labels_dir, filename)
            
            if os.path.exists(file_path):
                detection_class, view_class = self.parse_label_file(file_path)
                
                has_violation = False
                violation_type = None
                
                if detection_class is not None and view_class != -1:
                    has_violation = self.constraints.is_violation(view_class, detection_class)
                    if has_violation:
                        violation_type = self.constraints.get_violation_type(view_class, detection_class)
                
                return {
                    'exists': True,
                    'split': split,
                    'detection_class': detection_class,
                    'view_class': view_class,
                    'has_violation': has_violation,
                    'violation_type': violation_type,
                    'file_path': file_path
                }
        
        return {
            'exists': False,
            'split': None,
            'detection_class': None,
            'view_class': None,
            'has_violation': None,
            'violation_type': None,
            'file_path': None
        }
    
    def generate_report(self, investigations):
        """生成調查報告"""
        print("\n" + "=" * 80)
        print("違規調查摘要報告")
        print("=" * 80)
        
        # 統計分析
        total_violations = sum(len(inv) for inv in investigations.values())
        files_not_in_v1 = 0
        files_in_different_splits = 0
        files_same_split_but_violated = 0
        
        for dataset_name, dataset_investigations in investigations.items():
            for inv in dataset_investigations:
                v1_status = inv['v1_status']
                
                if not v1_status['exists']:
                    files_not_in_v1 += 1
                elif v1_status['split'] != inv['current_split']:
                    files_in_different_splits += 1
                else:
                    files_same_split_but_violated += 1
        
        print(f"總違規檔案: {total_violations}")
        print(f"在 V1 中不存在: {files_not_in_v1}")
        print(f"在 V1 不同分割中: {files_in_different_splits}")
        print(f"在 V1 同分割但仍違規: {files_same_split_but_violated}")
        
        # 詳細分析
        print(f"\n詳細分析:")
        
        if files_not_in_v1 > 0:
            print(f"\n{files_not_in_v1} 個檔案在 V1 中不存在:")
            count = 0
            for dataset_name, dataset_investigations in investigations.items():
                for inv in dataset_investigations:
                    if not inv['v1_status']['exists'] and count < 5:
                        print(f"  - {inv['filename']} (在 {dataset_name})")
                        count += 1
            if count >= 5:
                print(f"  ... 還有 {files_not_in_v1 - 5} 個檔案")
        
        if files_in_different_splits > 0:
            print(f"\n{files_in_different_splits} 個檔案在 V1 的不同分割中:")
            count = 0
            for dataset_name, dataset_investigations in investigations.items():
                for inv in dataset_investigations:
                    v1_status = inv['v1_status']
                    if (v1_status['exists'] and 
                        v1_status['split'] != inv['current_split'] and 
                        count < 5):
                        print(f"  - {inv['filename']}: {dataset_name}/{inv['current_split']} vs V1/{v1_status['split']}")
                        count += 1
            if count >= 5:
                print(f"  ... 還有 {files_in_different_splits - 5} 個檔案")
        
        if files_same_split_but_violated > 0:
            print(f"\n⚠️ {files_same_split_but_violated} 個檔案同步可能失敗:")
            for dataset_name, dataset_investigations in investigations.items():
                for inv in dataset_investigations:
                    v1_status = inv['v1_status']
                    if (v1_status['exists'] and 
                        v1_status['split'] == inv['current_split']):
                        print(f"  - {inv['filename']} 在 {dataset_name}/{inv['current_split']}")
        
        return {
            'total_violations': total_violations,
            'files_not_in_v1': files_not_in_v1,
            'files_in_different_splits': files_in_different_splits,
            'files_same_split_but_violated': files_same_split_but_violated,
            'detailed_investigations': investigations
        }


def main():
    """主函數"""
    print("=== V2-V5 剩餘違規調查 ===")
    print("調查同步後仍存在違規的原因")
    print("=" * 60)
    
    investigator = ViolationInvestigator()
    
    # 掃描檔案
    investigator.scan_all_files()
    
    # 調查違規
    investigations = investigator.investigate_violations()
    
    # 生成報告
    report = investigator.generate_report(investigations)
    
    print("\n" + "=" * 60)
    print("調查完成！")


if __name__ == "__main__":
    main()
