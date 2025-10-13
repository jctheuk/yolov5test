#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨分割違規修正腳本
將 V2-V5 中的違規檔案標籤修正為 V1 中對應檔案的正確標籤
保持 K-fold 分割結構不變
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime


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


class CrossSplitViolationFixer:
    """跨分割違規修正器"""
    
    def __init__(self):
        self.constraints = AnatomicalConstraints()
        self.datasets = {
            'V1': './regurgitationV1',
            'V2': './regurgitationV2', 
            'V3': './regurgitationV3',
            'V4': './regurgitationV4',
            'V5': './regurgitationV5'
        }
        
        self.v1_file_map = {}  # {filename: {split: path, label_content: content}}
        self.fixes_applied = []
        
    def build_v1_file_map(self):
        """建立 V1 檔案映射表"""
        print("建立 V1 檔案映射表...")
        
        v1_path = self.datasets['V1']
        
        for split in ['train', 'valid', 'test']:
            labels_dir = os.path.join(v1_path, split, 'labels')
            if not os.path.exists(labels_dir):
                continue
            
            for filename in os.listdir(labels_dir):
                if not filename.endswith('.txt'):
                    continue
                
                file_path = os.path.join(labels_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    self.v1_file_map[filename] = {
                        'split': split,
                        'path': file_path,
                        'content': content
                    }
                
                except Exception as e:
                    print(f"無法讀取 V1 檔案 {file_path}: {e}")
        
        print(f"V1 映射表建立完成，包含 {len(self.v1_file_map)} 個檔案")
    
    def parse_label_content(self, content):
        """解析標籤內容"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
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
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    detection_class, view_class = self.parse_label_content(content)
                    
                    if detection_class is None or view_class == -1:
                        continue
                    
                    if self.constraints.is_violation(view_class, detection_class):
                        violation_type = self.constraints.get_violation_type(view_class, detection_class)
                        violations.append({
                            'filename': filename,
                            'split': split,
                            'path': file_path,
                            'content': content,
                            'detection_class': detection_class,
                            'view_class': view_class,
                            'violation_type': violation_type
                        })
                
                except Exception as e:
                    continue
        
        return violations
    
    def fix_violations_in_dataset(self, dataset_name):
        """修正資料集中的違規"""
        print(f"\n=== 修正 {dataset_name} 中的違規 ===")
        
        violations = self.find_violations_in_dataset(dataset_name)
        print(f"發現 {len(violations)} 個違規")
        
        fixes_count = 0
        
        for violation in violations:
            filename = violation['filename']
            
            # 檢查 V1 中是否有這個檔案
            if filename not in self.v1_file_map:
                print(f"  ⚠️  {filename}: V1 中沒有對應檔案，跳過")
                continue
            
            v1_info = self.v1_file_map[filename]
            v1_content = v1_info['content']
            
            # 檢查 V1 的標籤是否正確（無違規）
            v1_detection_class, v1_view_class = self.parse_label_content(v1_content)
            
            if (v1_detection_class is None or v1_view_class == -1 or
                self.constraints.is_violation(v1_view_class, v1_detection_class)):
                print(f"  ⚠️  {filename}: V1 版本也有問題，跳過")
                continue
            
            # 應用修正
            try:
                # 備份原檔案
                backup_path = violation['path'] + '.backup'
                shutil.copy2(violation['path'], backup_path)
                
                # 寫入 V1 的正確內容
                with open(violation['path'], 'w', encoding='utf-8') as f:
                    f.write(v1_content)
                
                # 記錄修正
                fix_record = {
                    'dataset': dataset_name,
                    'filename': filename,
                    'target_split': violation['split'],
                    'source_split': v1_info['split'],
                    'original_violation': violation['violation_type'],
                    'original_content': violation['content'],
                    'fixed_content': v1_content,
                    'backup_path': backup_path
                }
                
                self.fixes_applied.append(fix_record)
                fixes_count += 1
                
                print(f"  ✅ {filename}: {violation['violation_type']} → 修正為 V1 版本 (來自 {v1_info['split']})")
                
            except Exception as e:
                print(f"  ❌ {filename}: 修正失敗 - {e}")
        
        print(f"  修正完成: {fixes_count}/{len(violations)} 個檔案")
        return fixes_count
    
    def fix_all_datasets(self):
        """修正所有目標資料集"""
        print("開始修正所有資料集的違規...")
        
        total_fixes = 0
        
        for dataset_name in ['V2', 'V3', 'V4', 'V5']:
            fixes = self.fix_violations_in_dataset(dataset_name)
            total_fixes += fixes
        
        print(f"\n總共修正了 {total_fixes} 個違規檔案")
        return total_fixes
    
    def verify_fixes(self):
        """驗證修正結果"""
        print("\n驗證修正結果...")
        
        verification_results = {}
        
        for dataset_name in ['V2', 'V3', 'V4', 'V5']:
            violations = self.find_violations_in_dataset(dataset_name)
            verification_results[dataset_name] = len(violations)
            print(f"  {dataset_name}: {len(violations)} 個剩餘違規")
        
        total_remaining = sum(verification_results.values())
        print(f"  總剩餘違規: {total_remaining}")
        
        return verification_results
    
    def generate_report(self, verification_results):
        """生成修正報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建輸出目錄
        output_dir = Path("./cross_split_fix_reports")
        output_dir.mkdir(exist_ok=True)
        
        # 準備報告資料
        report_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_fixes_applied': len(self.fixes_applied),
                'verification_results': verification_results
            },
            'fixes_applied': self.fixes_applied
        }
        
        # 生成 JSON 報告
        json_path = output_dir / f"cross_split_fix_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 報告
        md_content = self.create_markdown_report(report_data)
        md_path = output_dir / f"CROSS_SPLIT_FIX_REPORT_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n修正報告已生成:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        
        return str(md_path)
    
    def create_markdown_report(self, data):
        """創建 Markdown 格式的修正報告"""
        metadata = data['metadata']
        fixes = data['fixes_applied']
        
        total_remaining = sum(metadata['verification_results'].values())
        
        report = f"""# 跨分割違規修正報告

**時間**: {metadata['timestamp']}  
**修正檔案數**: {metadata['total_fixes_applied']}  
**剩餘違規數**: {total_remaining}

---

## 📊 修正統計

| 資料集 | 修正後剩餘違規 |
|--------|----------------|"""
        
        for dataset, remaining in metadata['verification_results'].items():
            report += f"\n| {dataset} | {remaining} |"
        
        report += f"""

總修正數: **{len(fixes)}** 個檔案

---

## ✅ 結果評估

"""
        
        if total_remaining == 0:
            report += """**🎉 完全成功！**

所有約束違規已修正完成，V2-V5 資料集現在都符合醫學解剖學約束。
"""
        elif total_remaining < 10:
            report += f"""**✅ 基本成功！**

只剩餘 {total_remaining} 個違規，大部分問題已解決。
"""
        else:
            report += f"""**⚠️ 部分成功**

還有 {total_remaining} 個違規需要進一步處理。
"""
        
        # 修正範例
        if fixes:
            report += """
---

## 🔧 修正範例

以下是一些修正的案例：

"""
            
            for i, fix in enumerate(fixes[:5], 1):
                report += f"""### {i}. {fix['filename']}
- **資料集**: {fix['dataset']}
- **目標分割**: {fix['target_split']}
- **來源分割**: {fix['source_split']} (V1)
- **原始違規**: {fix['original_violation']}
- **修正狀態**: ✅ 成功

"""
            
            if len(fixes) > 5:
                report += f"*（還有 {len(fixes) - 5} 個修正，詳見 JSON 檔案）*\n"
        
        report += """
---

## 📋 後續步驟

1. **重新檢查約束違規** - 確認所有資料集現在都乾淨
2. **清理快取檔案** - 刪除所有快取以確保使用新標籤
3. **開始訓練** - 現在可以使用任何資料集進行訓練

---

*修正完成時間: """ + metadata['timestamp'] + "*"

        return report


def main():
    """主函數"""
    print("=== 跨分割違規修正工具 ===")
    print("將 V2-V5 中的違規修正為 V1 的正確標籤")
    print("保持 K-fold 分割結構不變")
    print("=" * 60)
    
    # 初始化修正器
    fixer = CrossSplitViolationFixer()
    
    # 建立 V1 檔案映射
    fixer.build_v1_file_map()
    
    # 執行修正
    total_fixes = fixer.fix_all_datasets()
    
    if total_fixes == 0:
        print("\n沒有需要修正的違規檔案")
        return
    
    # 驗證修正結果
    verification_results = fixer.verify_fixes()
    
    # 生成報告
    report_path = fixer.generate_report(verification_results)
    
    # 顯示最終結果
    print("\n" + "=" * 60)
    print("修正完成!")
    
    total_remaining = sum(verification_results.values())
    
    if total_remaining == 0:
        print("🎉 所有違規已完全修正！")
    elif total_remaining < 10:
        print(f"✅ 修正基本成功，剩餘 {total_remaining} 個違規")
    else:
        print(f"⚠️ 修正部分成功，剩餘 {total_remaining} 個違規")
    
    print(f"\n📄 詳細報告: {report_path}")
    
    # 提醒後續步驟
    print("\n🔄 建議後續步驟:")
    print("1. 清理快取檔案")
    print("2. 重新檢查所有資料集的約束違規")
    print("3. 驗證修正效果")


if __name__ == "__main__":
    main()
