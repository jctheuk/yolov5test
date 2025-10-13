#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync V1 Labels to V2-V5 Datasets
將 V1 資料集的正確標籤同步到 V2-V5 資料集
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime


class LabelSynchronizer:
    """標籤同步器"""
    
    def __init__(self):
        self.datasets = {
            'V1': './regurgitationV1',
            'V2': './regurgitationV2', 
            'V3': './regurgitationV3',
            'V4': './regurgitationV4',
            'V5': './regurgitationV5'
        }
        
        self.source_dataset = 'V1'  # 源資料集（正確版本）
        self.target_datasets = ['V2', 'V3', 'V4', 'V5']  # 目標資料集
        
        self.sync_stats = {
            'total_files_found': 0,
            'files_synced': 0,
            'files_skipped': 0,
            'errors': [],
            'dataset_stats': {}
        }
    
    def find_all_label_files(self):
        """找到所有標籤檔案的對應關係"""
        print("掃描所有資料集的標籤檔案...")
        
        file_map = defaultdict(dict)  # {filename: {dataset: {split: path}}}
        
        for dataset_name, dataset_path in self.datasets.items():
            if not os.path.exists(dataset_path):
                print(f"警告: {dataset_path} 不存在")
                continue
            
            dataset_files = 0
            
            for split in ['train', 'valid', 'test']:
                labels_dir = os.path.join(dataset_path, split, 'labels')
                if not os.path.exists(labels_dir):
                    continue
                
                for filename in os.listdir(labels_dir):
                    if not filename.endswith('.txt'):
                        continue
                    
                    file_path = os.path.join(labels_dir, filename)
                    
                    if dataset_name not in file_map[filename]:
                        file_map[filename][dataset_name] = {}
                    
                    file_map[filename][dataset_name][split] = file_path
                    dataset_files += 1
            
            print(f"  {dataset_name}: {dataset_files} 個標籤檔案")
        
        return file_map
    
    def sync_labels(self, file_map):
        """同步標籤檔案"""
        print(f"\n開始將 {self.source_dataset} 的標籤同步到目標資料集...")
        
        synced_count = 0
        skipped_count = 0
        
        for filename, dataset_files in file_map.items():
            # 檢查源檔案是否存在
            if self.source_dataset not in dataset_files:
                skipped_count += 1
                continue
            
            source_files = dataset_files[self.source_dataset]
            
            # 為每個目標資料集同步
            for target_dataset in self.target_datasets:
                if target_dataset not in dataset_files:
                    continue
                
                target_files = dataset_files[target_dataset]
                
                # 同步每個分割
                for split in source_files.keys():
                    if split not in target_files:
                        continue
                    
                    source_path = source_files[split]
                    target_path = target_files[split]
                    
                    try:
                        # 複製標籤檔案
                        shutil.copy2(source_path, target_path)
                        synced_count += 1
                        
                    except Exception as e:
                        error_msg = f"錯誤同步 {filename} 到 {target_dataset}/{split}: {e}"
                        self.sync_stats['errors'].append(error_msg)
                        print(f"  {error_msg}")
        
        self.sync_stats['files_synced'] = synced_count
        self.sync_stats['files_skipped'] = skipped_count
        
        print(f"  成功同步: {synced_count} 個檔案")
        print(f"  跳過: {skipped_count} 個檔案")
        
        return synced_count > 0
    
    def verify_sync_result(self, file_map):
        """驗證同步結果"""
        print("\n驗證同步結果...")
        
        verification_stats = {
            'total_checked': 0,
            'identical_files': 0,
            'different_files': [],
            'missing_files': []
        }
        
        for filename, dataset_files in file_map.items():
            if self.source_dataset not in dataset_files:
                continue
            
            source_files = dataset_files[self.source_dataset]
            
            for split in source_files.keys():
                source_path = source_files[split]
                
                if not os.path.exists(source_path):
                    continue
                
                # 讀取源檔案內容
                try:
                    with open(source_path, 'r', encoding='utf-8') as f:
                        source_content = f.read().strip()
                except Exception as e:
                    continue
                
                # 檢查每個目標資料集
                for target_dataset in self.target_datasets:
                    if (target_dataset not in dataset_files or 
                        split not in dataset_files[target_dataset]):
                        verification_stats['missing_files'].append(f"{target_dataset}/{split}/{filename}")
                        continue
                    
                    target_path = dataset_files[target_dataset][split]
                    
                    try:
                        with open(target_path, 'r', encoding='utf-8') as f:
                            target_content = f.read().strip()
                        
                        verification_stats['total_checked'] += 1
                        
                        if source_content == target_content:
                            verification_stats['identical_files'] += 1
                        else:
                            verification_stats['different_files'].append(f"{target_dataset}/{split}/{filename}")
                    
                    except Exception as e:
                        verification_stats['missing_files'].append(f"{target_dataset}/{split}/{filename}")
        
        # 顯示驗證結果
        total_checked = verification_stats['total_checked']
        identical = verification_stats['identical_files']
        different_count = len(verification_stats['different_files'])
        missing_count = len(verification_stats['missing_files'])
        
        print(f"  檢查的檔案對: {total_checked}")
        print(f"  完全相同: {identical} ({identical/total_checked*100:.1f}%)")
        print(f"  不同內容: {different_count}")
        print(f"  缺失檔案: {missing_count}")
        
        if different_count > 0:
            print(f"  前幾個不同的檔案:")
            for diff_file in verification_stats['different_files'][:5]:
                print(f"    - {diff_file}")
        
        return verification_stats
    
    def generate_sync_report(self, verification_stats):
        """生成同步報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建輸出目錄
        output_dir = Path("./label_sync_reports")
        output_dir.mkdir(exist_ok=True)
        
        # 準備報告資料
        report_data = {
            'metadata': {
                'timestamp': timestamp,
                'source_dataset': self.source_dataset,
                'target_datasets': self.target_datasets,
                'sync_stats': self.sync_stats,
                'verification_stats': verification_stats
            }
        }
        
        # 生成 JSON 報告
        json_path = output_dir / f"label_sync_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 報告
        md_content = self.create_markdown_report(report_data)
        md_path = output_dir / f"LABEL_SYNC_REPORT_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n同步報告已生成:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        
        return str(md_path)
    
    def create_markdown_report(self, data):
        """創建 Markdown 格式的同步報告"""
        metadata = data['metadata']
        sync_stats = metadata['sync_stats']
        verification_stats = metadata['verification_stats']
        
        success_rate = (verification_stats['identical_files'] / 
                       verification_stats['total_checked'] * 100 
                       if verification_stats['total_checked'] > 0 else 0)
        
        report = f"""# V1 標籤同步報告

**時間**: {metadata['timestamp']}  
**源資料集**: {metadata['source_dataset']} (已修正的正確版本)  
**目標資料集**: {', '.join(metadata['target_datasets'])}

---

## 📊 同步統計

| 項目 | 數量 | 說明 |
|------|------|------|
| 同步檔案 | {sync_stats['files_synced']} | 成功從 V1 複製的標籤檔案 |
| 跳過檔案 | {sync_stats['files_skipped']} | 源資料集中不存在的檔案 |
| 錯誤數 | {len(sync_stats['errors'])} | 同步過程中的錯誤 |

---

## ✅ 驗證結果

| 項目 | 數量 | 比例 |
|------|------|------|
| 檢查檔案對 | {verification_stats['total_checked']} | 總驗證數 |
| 完全相同 | {verification_stats['identical_files']} | {success_rate:.1f}% |
| 內容不同 | {len(verification_stats['different_files'])} | - |
| 缺失檔案 | {len(verification_stats['missing_files'])} | - |

---

## 🎯 結果評估

"""

        if success_rate >= 99.9:
            report += """**✅ 同步完全成功！**

所有標籤檔案已成功從 V1 同步到 V2-V5，現在所有資料集都使用相同的正確標籤。
"""
        elif success_rate >= 95:
            report += f"""**✅ 同步基本成功！**

{success_rate:.1f}% 的檔案已成功同步，少數檔案可能需要手動檢查。
"""
        else:
            report += f"""**⚠️ 同步部分成功**

只有 {success_rate:.1f}% 的檔案成功同步，需要檢查同步過程中的問題。
"""

        # 錯誤詳情
        if sync_stats['errors']:
            report += f"""
---

## ⚠️ 同步錯誤

同步過程中遇到 {len(sync_stats['errors'])} 個錯誤：

"""
            for i, error in enumerate(sync_stats['errors'][:10], 1):
                report += f"{i}. {error}\n"
            
            if len(sync_stats['errors']) > 10:
                report += f"\n*（還有 {len(sync_stats['errors']) - 10} 個錯誤，詳見 JSON 檔案）*\n"

        # 不同檔案詳情
        if verification_stats['different_files']:
            report += f"""
---

## 🔍 內容不同的檔案

發現 {len(verification_stats['different_files'])} 個檔案內容不同：

"""
            for i, diff_file in enumerate(verification_stats['different_files'][:10], 1):
                report += f"{i}. {diff_file}\n"
            
            if len(verification_stats['different_files']) > 10:
                report += f"\n*（還有 {len(verification_stats['different_files']) - 10} 個不同檔案，詳見 JSON 檔案）*\n"

        report += """
---

## 📋 後續步驟

1. **重新檢查約束違規** - 驗證所有資料集現在都沒有違規
2. **清理快取檔案** - 刪除所有快取以確保使用新標籤
3. **開始訓練** - 現在可以使用任何資料集進行訓練

---

*報告生成時間: """ + metadata['timestamp'] + "*"

        return report


def main():
    """主函數"""
    print("=== V1 標籤同步到 V2-V5 資料集 ===")
    print("將 V1 的正確標籤複製到 V2-V5 資料集")
    print("=" * 60)
    
    # 初始化同步器
    synchronizer = LabelSynchronizer()
    
    # 掃描所有標籤檔案
    file_map = synchronizer.find_all_label_files()
    
    print(f"\n找到 {len(file_map)} 個不同的標籤檔案")
    
    # 確認操作
    print(f"\n即將將 {synchronizer.source_dataset} 的標籤同步到:")
    for target in synchronizer.target_datasets:
        print(f"  - {target}")
    
    print("\n✅ 自動執行同步...")
    
    # 執行同步
    success = synchronizer.sync_labels(file_map)
    
    if not success:
        print("❌ 同步失敗!")
        return
    
    # 驗證結果
    verification_stats = synchronizer.verify_sync_result(file_map)
    
    # 生成報告
    report_path = synchronizer.generate_sync_report(verification_stats)
    
    # 顯示最終結果
    print("\n" + "=" * 60)
    print("同步完成!")
    
    total_checked = verification_stats['total_checked']
    identical = verification_stats['identical_files']
    
    if total_checked > 0:
        success_rate = identical / total_checked * 100
        
        if success_rate >= 99.9:
            print("✅ 所有標籤已成功同步!")
        elif success_rate >= 95:
            print(f"✅ 基本同步成功 ({success_rate:.1f}%)")
        else:
            print(f"⚠️ 部分同步成功 ({success_rate:.1f}%)")
    
    print(f"\n📄 詳細報告: {report_path}")
    
    # 提醒後續步驟
    print("\n🔄 建議後續步驟:")
    print("1. 清理快取檔案")  
    print("2. 重新檢查約束違規")
    print("3. 驗證所有資料集現在都相同")


if __name__ == "__main__":
    main()
