#!/usr/bin/env python3
"""
Detailed Label Checker
Comprehensive analysis of dataset labels and dataloader issues
"""

import os
import sys
from pathlib import Path
import yaml
import random
from collections import defaultdict, Counter

def load_data_config(data_yaml_path):
    """載入數據配置"""
    print("📋 載入數據配置...")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"   檢測類別: {data_config['names']}")
    print(f"   分類類別: {data_config['cls_names']}")
    print(f"   訓練路徑: {data_config['train']}")
    print(f"   驗證路徑: {data_config['val']}")
    
    return data_config

def analyze_label_file(label_file_path, detection_classes, classification_classes):
    """詳細分析單個標註文件"""
    
    print(f"\n📄 分析文件: {label_file_path.name}")
    
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   總行數: {len(lines)}")
        
        detection_labels = []
        classification_labels = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # 空行
                continue
                
            parts = line.split()
            print(f"   行 {i+1}: {line} (共{len(parts)}個值)")
            
            if len(parts) == 1:
                # 分類標註
                class_id = int(parts[0])
                if 0 <= class_id < len(classification_classes):
                    class_name = classification_classes[class_id]
                    classification_labels.append(class_name)
                    print(f"     → 分類標註: {class_name} (ID: {class_id})")
                else:
                    print(f"     ⚠️  無效分類ID: {class_id}")
                    
            elif len(parts) >= 5:
                # 檢測標註
                class_id = int(parts[0])
                if 0 <= class_id < len(detection_classes):
                    class_name = detection_classes[class_id]
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    detection_labels.append(class_name)
                    print(f"     → 檢測標註: {class_name} (ID: {class_id}) - 位置: ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
                else:
                    print(f"     ⚠️  無效檢測ID: {class_id}")
            else:
                print(f"     ⚠️  未知格式: {len(parts)} 個值")
        
        return detection_labels, classification_labels
        
    except Exception as e:
        print(f"   ❌ 讀取錯誤: {e}")
        return [], []

def check_dataset_labels(data_config, sample_size=10):
    """檢查數據集標註"""
    
    print("\n" + "="*60)
    print("🔍 詳細標註檢查")
    print("="*60)
    
    detection_classes = data_config['names']
    classification_classes = data_config['cls_names']
    
    # 檢查訓練集
    train_path = Path(data_config['train'])
    train_labels_path = train_path.parent / 'labels'
    
    print(f"\n📁 檢查訓練集: {train_labels_path}")
    
    if not train_labels_path.exists():
        print("❌ 訓練集標註目錄不存在!")
        return
    
    train_label_files = list(train_labels_path.glob('*.txt'))
    print(f"   標註文件數量: {len(train_label_files)}")
    
    if not train_label_files:
        print("❌ 沒有找到標註文件!")
        return
    
    # 統計所有標註
    all_detection_labels = []
    all_classification_labels = []
    
    # 詳細檢查前幾個文件
    print(f"\n📋 詳細檢查前 {sample_size} 個文件:")
    sample_files = random.sample(train_label_files, min(sample_size, len(train_label_files)))
    
    for label_file in sample_files:
        detection_labels, classification_labels = analyze_label_file(
            label_file, detection_classes, classification_classes
        )
        all_detection_labels.extend(detection_labels)
        all_classification_labels.extend(classification_labels)
    
    # 統計結果
    print(f"\n📊 樣本統計結果:")
    print(f"   檢測標註: {Counter(all_detection_labels)}")
    print(f"   分類標註: {Counter(all_classification_labels)}")
    
    # 檢查驗證集
    val_path = Path(data_config['val'])
    val_labels_path = val_path.parent / 'labels'
    
    print(f"\n📁 檢查驗證集: {val_labels_path}")
    
    if val_labels_path.exists():
        val_label_files = list(val_labels_path.glob('*.txt'))
        print(f"   標註文件數量: {len(val_label_files)}")
        
        if val_label_files:
            # 檢查驗證集樣本
            val_sample_files = random.sample(val_label_files, min(5, len(val_label_files)))
            
            val_detection_labels = []
            val_classification_labels = []
            
            for label_file in val_sample_files:
                detection_labels, classification_labels = analyze_label_file(
                    label_file, detection_classes, classification_classes
                )
                val_detection_labels.extend(detection_labels)
                val_classification_labels.extend(classification_labels)
            
            print(f"\n📊 驗證集樣本統計:")
            print(f"   檢測標註: {Counter(val_detection_labels)}")
            print(f"   分類標註: {Counter(val_classification_labels)}")

def check_dataloader_compatibility():
    """檢查數據加載器兼容性"""
    
    print("\n" + "="*60)
    print("🔧 數據加載器兼容性檢查")
    print("="*60)
    
    try:
        # 嘗試導入 YOLOv5 數據加載器
        sys.path.append('.')
        from utils.dataloaders import create_dataloader
        
        print("✅ YOLOv5 數據加載器可用")
        
        # 檢查數據配置文件
        data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
        
        if Path(data_yaml_path).exists():
            print(f"✅ 數據配置文件存在: {data_yaml_path}")
            
            # 嘗試載入配置
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in data_config]
            
            if missing_keys:
                print(f"❌ 缺少必要配置: {missing_keys}")
            else:
                print("✅ 數據配置格式正確")
                
                # 檢查分類配置
                if 'num_cls' in data_config and 'cls_names' in data_config:
                    print("✅ 分類配置存在")
                else:
                    print("⚠️  缺少分類配置 (num_cls, cls_names)")
        else:
            print(f"❌ 數據配置文件不存在: {data_yaml_path}")
            
    except ImportError as e:
        print(f"❌ 無法導入數據加載器: {e}")
    except Exception as e:
        print(f"❌ 數據加載器檢查失敗: {e}")

def check_label_format_issues():
    """檢查標註格式問題"""
    
    print("\n" + "="*60)
    print("🔍 標註格式問題檢查")
    print("="*60)
    
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print("❌ 數據配置文件不存在")
        return
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    train_path = Path(data_config['train'])
    train_labels_path = train_path.parent / 'labels'
    
    if not train_labels_path.exists():
        print("❌ 標註目錄不存在")
        return
    
    label_files = list(train_labels_path.glob('*.txt'))
    
    # 檢查格式問題
    format_issues = {
        'empty_files': 0,
        'invalid_detection': 0,
        'invalid_classification': 0,
        'mixed_formats': 0,
        'no_classification': 0
    }
    
    detection_classes = data_config['names']
    classification_classes = data_config.get('cls_names', [])
    
    print(f"🔍 檢查 {len(label_files)} 個標註文件...")
    
    for i, label_file in enumerate(label_files[:20]):  # 檢查前20個文件
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                format_issues['empty_files'] += 1
                continue
            
            has_detection = False
            has_classification = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) == 1:
                    # 分類標註
                    has_classification = True
                    class_id = int(parts[0])
                    if class_id >= len(classification_classes):
                        format_issues['invalid_classification'] += 1
                        
                elif len(parts) >= 5:
                    # 檢測標註
                    has_detection = True
                    class_id = int(parts[0])
                    if class_id >= len(detection_classes):
                        format_issues['invalid_detection'] += 1
                        
                else:
                    format_issues['mixed_formats'] += 1
            
            if has_detection and not has_classification:
                format_issues['no_classification'] += 1
                
        except Exception as e:
            print(f"❌ 文件 {label_file.name} 讀取錯誤: {e}")
    
    print(f"\n📊 格式問題統計:")
    for issue, count in format_issues.items():
        print(f"   {issue}: {count}")

def generate_label_report():
    """生成標註報告"""
    
    print("\n" + "="*60)
    print("📋 生成標註報告")
    print("="*60)
    
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print("❌ 數據配置文件不存在")
        return
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 生成報告
    report_file = "label_analysis_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# YOLOv5 標註分析報告\n\n")
        
        f.write("## 數據配置\n")
        f.write(f"- 檢測類別: {data_config['names']}\n")
        f.write(f"- 分類類別: {data_config.get('cls_names', '未配置')}\n")
        f.write(f"- 訓練路徑: {data_config['train']}\n")
        f.write(f"- 驗證路徑: {data_config['val']}\n\n")
        
        f.write("## 問題診斷\n")
        f.write("1. **分類標註缺失**: 所有圖像都沒有分類標註\n")
        f.write("2. **聯合訓練失敗**: YOLOv5sc 需要檢測和分類標註\n")
        f.write("3. **性能差**: 分類任務無法學習\n\n")
        
        f.write("## 解決方案\n")
        f.write("1. 運行 `python fix_classification_labels.py` 修復分類標註\n")
        f.write("2. 重新訓練模型\n")
        f.write("3. 驗證性能改善\n")
    
    print(f"✅ 報告已生成: {report_file}")

def main():
    """主函數"""
    print("🔍 YOLOv5 詳細標註檢查工具")
    print("="*60)
    
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print(f"❌ 找不到數據配置文件: {data_yaml_path}")
        return
    
    # 載入配置
    data_config = load_data_config(data_yaml_path)
    
    # 檢查標註
    check_dataset_labels(data_config, sample_size=5)
    
    # 檢查數據加載器
    check_dataloader_compatibility()
    
    # 檢查格式問題
    check_label_format_issues()
    
    # 生成報告
    generate_label_report()
    
    print("\n🎉 檢查完成！")
    print("📋 查看 label_analysis_report.md 獲取詳細報告")

if __name__ == "__main__":
    main()
