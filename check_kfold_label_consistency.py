#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Fold Label Consistency Checker for V1-V5 Datasets
檢查 V1-V5 資料集中相同檔案的標籤一致性
考慮到 V1 中已修正的違規情況
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime


class AnatomicalConstraints:
    """解剖學約束定義（用於識別已知的違規修正）"""
    
    def __init__(self):
        self.view_names = {0: 'A4C', 1: 'PSAX', 2: 'PLAX'}
        self.regurg_names = {0: 'AR', 1: 'MR', 2: 'PR', 3: 'TR'}
        
        # 約束規則：每個視圖允許的反流類型
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


class KFoldConsistencyChecker:
    """K-Fold 資料集標籤一致性檢查器"""
    
    def __init__(self):
        self.constraints = AnatomicalConstraints()
        self.datasets = {
            'V1': './regurgitationV1',
            'V2': './regurgitationV2', 
            'V3': './regurgitationV3',
            'V4': './regurgitationV4',
            'V5': './regurgitationV5'
        }
        
        # 儲存所有檔案的標籤資料
        self.all_labels = {}  # {filename: {dataset: {detection_data, classification_data}}}
        self.inconsistencies = []
        self.known_v1_fixes = []  # V1 中的已知違規修正
        
    def parse_label_file(self, file_path):
        """解析標籤檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) != 2:
                return None, None
            
            # 解析檢測資料 (第一行)
            detection_parts = lines[0].split()
            if len(detection_parts) != 5:
                return None, None
            
            detection_data = {
                'class': int(detection_parts[0]),
                'x': float(detection_parts[1]), 
                'y': float(detection_parts[2]),
                'w': float(detection_parts[3]),
                'h': float(detection_parts[4])
            }
            
            # 解析分類資料 (第二行)
            classification_parts = lines[1].split()
            if len(classification_parts) != 3:
                return None, None
                
            classification_data = [int(p) for p in classification_parts]
            view_class = classification_data.index(1) if 1 in classification_data else -1
            
            return detection_data, {'raw': classification_data, 'view_class': view_class}
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None, None
    
    def collect_all_labels(self):
        """收集所有資料集的標籤資料"""
        print("收集所有資料集的標籤資料...")
        
        for dataset_name, dataset_path in self.datasets.items():
            if not os.path.exists(dataset_path):
                print(f"警告: {dataset_path} 不存在")
                continue
            
            print(f"  處理 {dataset_name}...")
            dataset_files = 0
            
            # 處理所有分割
            for split in ['train', 'valid', 'test']:
                labels_dir = os.path.join(dataset_path, split, 'labels')
                if not os.path.exists(labels_dir):
                    continue
                
                for filename in os.listdir(labels_dir):
                    if not filename.endswith('.txt'):
                        continue
                    
                    file_path = os.path.join(labels_dir, filename)
                    detection_data, classification_data = self.parse_label_file(file_path)
                    
                    if detection_data is None:
                        continue
                    
                    # 初始化檔案記錄
                    if filename not in self.all_labels:
                        self.all_labels[filename] = {}
                    
                    self.all_labels[filename][dataset_name] = {
                        'detection': detection_data,
                        'classification': classification_data,
                        'split': split,
                        'path': file_path
                    }
                    
                    dataset_files += 1
            
            print(f"    {dataset_name}: {dataset_files} 個檔案")
    
    def find_inconsistencies(self):
        """找出標籤不一致的情況"""
        print("\n檢查標籤一致性...")
        
        total_files = len(self.all_labels)
        files_in_multiple_datasets = 0
        inconsistent_files = 0
        
        for filename, dataset_labels in self.all_labels.items():
            # 只檢查出現在多個資料集中的檔案
            if len(dataset_labels) <= 1:
                continue
            
            files_in_multiple_datasets += 1
            
            # 取得 V2 作為基準（假設 V2-V5 是原始資料）
            reference_dataset = None
            reference_data = None
            
            for dataset in ['V2', 'V3', 'V4', 'V5']:
                if dataset in dataset_labels:
                    reference_dataset = dataset
                    reference_data = dataset_labels[dataset]
                    break
            
            if reference_data is None:
                continue
            
            # 檢查與其他資料集的差異
            has_inconsistency = False
            inconsistency_details = {
                'filename': filename,
                'reference_dataset': reference_dataset,
                'differences': [],
                'datasets_involved': list(dataset_labels.keys())
            }
            
            for dataset_name, data in dataset_labels.items():
                if dataset_name == reference_dataset:
                    continue
                
                differences = []
                
                # 比較檢測資料
                ref_det = reference_data['detection']
                cur_det = data['detection']
                
                if ref_det['class'] != cur_det['class']:
                    differences.append(f"detection_class: {ref_det['class']} vs {cur_det['class']}")
                
                # 比較座標（容許小誤差）
                for coord in ['x', 'y', 'w', 'h']:
                    if abs(ref_det[coord] - cur_det[coord]) > 0.000001:
                        differences.append(f"detection_{coord}: {ref_det[coord]} vs {cur_det[coord]}")
                
                # 比較分類資料
                ref_cls = reference_data['classification']['raw']
                cur_cls = data['classification']['raw']
                
                if ref_cls != cur_cls:
                    differences.append(f"classification: {ref_cls} vs {cur_cls}")
                
                if differences:
                    # 檢查是否為已知的 V1 違規修正
                    is_known_v1_fix = False
                    if dataset_name == 'V1':
                        # 檢查是否為約束違規修正
                        ref_view_class = reference_data['classification']['view_class']
                        cur_detection_class = cur_det['class']
                        ref_detection_class = ref_det['class']
                        
                        if (self.constraints.is_violation(ref_view_class, ref_detection_class) and 
                            not self.constraints.is_violation(ref_view_class, cur_detection_class)):
                            is_known_v1_fix = True
                            self.known_v1_fixes.append({
                                'filename': filename,
                                'original_detection': ref_detection_class,
                                'fixed_detection': cur_detection_class,
                                'view_class': ref_view_class,
                                'violation_type': self.constraints.get_violation_type(ref_view_class, ref_detection_class)
                            })
                    
                    inconsistency_details['differences'].append({
                        'dataset': dataset_name,
                        'differences': differences,
                        'is_known_v1_fix': is_known_v1_fix
                    })
                    
                    if not is_known_v1_fix:
                        has_inconsistency = True
            
            if inconsistency_details['differences']:
                self.inconsistencies.append(inconsistency_details)
                if has_inconsistency:
                    inconsistent_files += 1
        
        print(f"  總檔案數: {total_files}")
        print(f"  出現在多個資料集的檔案: {files_in_multiple_datasets}")
        print(f"  有標籤差異的檔案: {len(self.inconsistencies)}")
        print(f"  未知不一致的檔案: {inconsistent_files}")
        print(f"  已知 V1 違規修正: {len(self.known_v1_fixes)}")
        
        return inconsistent_files == 0
    
    def generate_report(self):
        """生成詳細報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建輸出目錄
        output_dir = Path("./kfold_consistency_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # 生成 JSON 詳細資料
        detailed_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_files_checked': len(self.all_labels),
                'files_with_differences': len(self.inconsistencies),
                'known_v1_fixes': len(self.known_v1_fixes)
            },
            'inconsistencies': self.inconsistencies,
            'known_v1_fixes': self.known_v1_fixes
        }
        
        json_path = output_dir / f"kfold_consistency_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 報告
        report_content = self.create_markdown_report(detailed_data)
        
        md_path = output_dir / f"KFOLD_CONSISTENCY_REPORT_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n報告已生成:")
        print(f"  JSON 詳細資料: {json_path}")
        print(f"  Markdown 報告: {md_path}")
        
        return str(md_path)
    
    def create_markdown_report(self, data):
        """創建 Markdown 格式報告"""
        report = f"""# K-Fold 資料集標籤一致性分析報告

**生成時間**: {data['metadata']['timestamp']}
**分析對象**: regurgitationV1 到 regurgitationV5

---

## 📊 分析摘要

| 項目 | 數量 | 說明 |
|------|------|------|
| 總檢查檔案 | {data['metadata']['total_files_checked']} | 所有資料集中的標籤檔案 |
| 有差異的檔案 | {data['metadata']['files_with_differences']} | 在不同資料集版本間有標籤差異 |
| 已知 V1 修正 | {data['metadata']['known_v1_fixes']} | V1 中的已知約束違規修正 |

---

## ✅ 結論

"""
        
        unknown_inconsistencies = 0
        for inconsistency in data['inconsistencies']:
            has_unknown = any(not diff.get('is_known_v1_fix', False) 
                             for diff in inconsistency['differences'])
            if has_unknown:
                unknown_inconsistencies += 1
        
        if unknown_inconsistencies == 0:
            report += """**✅ 標籤一致性良好！**

除了 V1 中的已知違規修正外，所有資料集的標籤都保持一致。這表明 K-Fold 分割正確，相同檔案在不同資料集版本中的標籤是一致的。
"""
        else:
            report += f"""**⚠️ 發現 {unknown_inconsistencies} 個未知的標籤不一致！**

除了已知的 V1 違規修正外，還有其他標籤不一致的情況需要檢查。
"""
        
        # V1 修正摘要
        if data['known_v1_fixes']:
            report += f"""
---

## 🔧 V1 已知違規修正摘要

V1 中修正了 **{len(data['known_v1_fixes'])}** 個約束違規：

"""
            
            # 統計違規類型
            violation_types = {}
            for fix in data['known_v1_fixes']:
                vtype = fix['violation_type']
                if vtype not in violation_types:
                    violation_types[vtype] = []
                violation_types[vtype].append(fix['filename'])
            
            for vtype, filenames in violation_types.items():
                report += f"### {vtype}\n"
                report += f"- 修正數量: {len(filenames)}\n"
                report += f"- 檔案範例: `{filenames[0]}`\n\n"
        
        # 未知不一致詳情
        if unknown_inconsistencies > 0:
            report += """
---

## ⚠️ 未知標籤不一致詳情

以下檔案存在非 V1 違規修正的其他標籤不一致：

"""
            
            count = 0
            for inconsistency in data['inconsistencies']:
                unknown_diffs = [diff for diff in inconsistency['differences'] 
                               if not diff.get('is_known_v1_fix', False)]
                
                if unknown_diffs and count < 10:  # 只顯示前 10 個
                    report += f"### {inconsistency['filename']}\n"
                    report += f"- **參考資料集**: {inconsistency['reference_dataset']}\n"
                    report += f"- **涉及資料集**: {', '.join(inconsistency['datasets_involved'])}\n"
                    
                    for diff in unknown_diffs:
                        report += f"- **{diff['dataset']}**: {', '.join(diff['differences'])}\n"
                    
                    report += "\n"
                    count += 1
            
            if unknown_inconsistencies > 10:
                report += f"*（還有 {unknown_inconsistencies - 10} 個不一致情況，詳見 JSON 檔案）*\n"
        
        report += """
---

## 📋 建議

1. **V1 修正是正確的**：已知的約束違規修正符合醫學解剖學約束
2. **繼續使用 V1 進行訓練**：V1 是最乾淨的版本
3. **如有未知不一致**：需要進一步調查原因
"""
        
        return report


def main():
    """主函數"""
    print("=== K-Fold 資料集標籤一致性檢查 ===")
    print("檢查 V1-V5 資料集中相同檔案的標籤一致性")
    print("（考慮 V1 中的已知違規修正）")
    print("=" * 60)
    
    # 初始化檢查器
    checker = KFoldConsistencyChecker()
    
    # 收集資料
    checker.collect_all_labels()
    
    # 檢查一致性
    is_consistent = checker.find_inconsistencies()
    
    # 生成報告
    report_path = checker.generate_report()
    
    # 顯示結果
    print("\n" + "=" * 60)
    print("檢查完成！")
    
    if is_consistent:
        print("✅ 除了 V1 的已知違規修正外，所有標籤都保持一致！")
    else:
        print("⚠️  發現一些未知的標籤不一致，請檢查報告。")
    
    print(f"\n📄 詳細報告: {report_path}")


if __name__ == "__main__":
    main()
