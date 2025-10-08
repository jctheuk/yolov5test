#!/usr/bin/env python3
"""
檢測性能問題調查 - 數據質量檢查 (最終版)
檢查標註質量和一致性
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def load_data_config():
    """加載數據配置"""
    try:
        with open('regurgitationV1/data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config
    except Exception as e:
        print(f"❌ 無法加載 data.yaml: {e}")
        return None

def check_annotation_files(data_config):
    """檢查標註文件存在性和格式"""
    print("🔍 檢查標註文件...")
    
    issues = []
    total_files = 0
    valid_files = 0
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = data_config[split]
        # 修正路徑：從相對路徑轉為絕對路徑
        if split_path.startswith('../'):
            # 從 yolov5c 目錄的相對路徑轉為當前目錄的絕對路徑
            labels_path = split_path.replace('../regurgitationV1/', 'regurgitationV1/').replace('/images', '/labels')
        else:
            labels_path = split_path.replace('/images', '/labels')
        
        print(f"   檢查 {split}: {labels_path}")
        
        if not os.path.exists(labels_path):
            issues.append(f"❌ {split} labels 目錄不存在: {labels_path}")
            continue
            
        # 檢查標註文件
        label_files = list(Path(labels_path).glob('*.txt'))
        total_files += len(label_files)
        print(f"   找到 {len(label_files)} 個標註文件")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 檢查文件格式
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"❌ {label_file}:{line_num} 格式錯誤 (應為5個值): {line}")
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        # 檢查坐標範圍
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            issues.append(f"❌ {label_file}:{line_num} 坐標超出範圍: x={x}, y={y}, w={w}, h={h}")
                            continue
                            
                        # 檢查類別ID
                        if class_id < 0 or class_id >= 4:
                            issues.append(f"❌ {label_file}:{line_num} 類別ID無效: {class_id}")
                            continue
                            
                    except ValueError as e:
                        issues.append(f"❌ {label_file}:{line_num} 數值轉換錯誤: {line}")
                        continue
                
                valid_files += 1
                
            except Exception as e:
                issues.append(f"❌ 無法讀取 {label_file}: {e}")
    
    print(f"📊 標註文件統計:")
    print(f"   總文件數: {total_files}")
    print(f"   有效文件數: {valid_files}")
    print(f"   問題文件數: {total_files - valid_files}")
    
    if issues:
        print(f"\n⚠️ 發現 {len(issues)} 個問題:")
        for issue in issues[:10]:  # 只顯示前10個問題
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... 還有 {len(issues) - 10} 個問題")
    else:
        print("✅ 標註文件格式正確")
    
    return issues

def analyze_class_distribution(data_config):
    """分析類別分佈"""
    print("\n🔍 分析類別分佈...")
    
    class_counts = defaultdict(int)
    total_objects = 0
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = data_config[split]
        # 修正路徑：從相對路徑轉為絕對路徑
        if split_path.startswith('../'):
            labels_path = split_path.replace('../regurgitationV1/', 'regurgitationV1/').replace('/images', '/labels')
        else:
            labels_path = split_path.replace('/images', '/labels')
        
        if not os.path.exists(labels_path):
            continue
            
        print(f"   分析 {split}: {labels_path}")
        
        for label_file in Path(labels_path).glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_objects += 1
                        except ValueError:
                            continue
                            
            except Exception:
                continue
    
    print(f"📊 類別分佈統計:")
    class_names = ['AR', 'MR', 'PR', 'TR']
    for class_id in range(4):
        count = class_counts[class_id]
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"   {class_names[class_id]}: {count} ({percentage:.1f}%)")
    
    # 檢查類別不平衡
    imbalance_ratio = 1.0
    if total_objects > 0:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n⚠️ 類別不平衡分析:")
        print(f"   最大類別: {max_count}")
        print(f"   最小類別: {min_count}")
        print(f"   不平衡比例: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print("   🚨 嚴重類別不平衡 (比例 > 3)")
        elif imbalance_ratio > 2:
            print("   ⚠️ 中等類別不平衡 (比例 > 2)")
        else:
            print("   ✅ 類別分佈相對平衡")
    else:
        print("   ❌ 沒有找到有效的標註數據")
    
    return class_counts, total_objects, imbalance_ratio

def analyze_object_sizes(data_config):
    """分析目標尺寸分佈"""
    print("\n🔍 分析目標尺寸分佈...")
    
    sizes = []
    areas = []
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = data_config[split]
        # 修正路徑：從相對路徑轉為絕對路徑
        if split_path.startswith('../'):
            labels_path = split_path.replace('../regurgitationV1/', 'regurgitationV1/').replace('/images', '/labels')
        else:
            labels_path = split_path.replace('/images', '/labels')
        
        if not os.path.exists(labels_path):
            continue
            
        for label_file in Path(labels_path).glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            x, y, w, h = map(float, parts[1:5])
                            area = w * h
                            sizes.append((w, h))
                            areas.append(area)
                        except ValueError:
                            continue
                            
            except Exception:
                continue
    
    if areas:
        areas = np.array(areas)
        sizes = np.array(sizes)
        
        print(f"📊 目標尺寸統計:")
        print(f"   總目標數: {len(areas)}")
        print(f"   平均面積: {np.mean(areas):.4f}")
        print(f"   面積中位數: {np.median(areas):.4f}")
        print(f"   最小面積: {np.min(areas):.4f}")
        print(f"   最大面積: {np.max(areas):.4f}")
        print(f"   面積標準差: {np.std(areas):.4f}")
        
        # 分析小目標比例
        small_objects = np.sum(areas < 0.01)  # 面積 < 1%
        medium_objects = np.sum((areas >= 0.01) & (areas < 0.1))  # 1% <= 面積 < 10%
        large_objects = np.sum(areas >= 0.1)  # 面積 >= 10%
        
        print(f"\n📏 目標尺寸分佈:")
        print(f"   小目標 (< 1%): {small_objects} ({small_objects/len(areas)*100:.1f}%)")
        print(f"   中等目標 (1-10%): {medium_objects} ({medium_objects/len(areas)*100:.1f}%)")
        print(f"   大目標 (>= 10%): {large_objects} ({large_objects/len(areas)*100:.1f}%)")
        
        # 檢查小目標問題
        small_ratio = small_objects / len(areas)
        if small_ratio > 0.3:
            print("   🚨 小目標比例過高 (>30%) - 可能影響檢測性能")
        elif small_ratio > 0.2:
            print("   ⚠️ 小目標比例較高 (>20%) - 需要注意")
        else:
            print("   ✅ 小目標比例正常")
    else:
        small_ratio = 0.0
        print("❌ 沒有找到有效的目標數據")
    
    return sizes, areas, small_ratio

def check_image_quality(data_config):
    """檢查圖像質量"""
    print("\n🔍 檢查圖像質量...")
    
    image_stats = []
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = data_config[split]
        # 修正路徑：從相對路徑轉為絕對路徑
        if split_path.startswith('../'):
            images_path = split_path.replace('../regurgitationV1/', 'regurgitationV1/')
        else:
            images_path = split_path
        
        if not os.path.exists(images_path):
            continue
            
        print(f"   檢查 {split}: {images_path}")
        
        # 檢查前10張圖像
        image_files = list(Path(images_path).glob('*.png'))[:10]
        print(f"   找到 {len(image_files)} 張圖像")
        
        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w, c = img.shape
                    image_stats.append({
                        'file': img_file.name,
                        'width': w,
                        'height': h,
                        'channels': c,
                        'size': os.path.getsize(img_file)
                    })
            except Exception as e:
                print(f"❌ 無法讀取圖像 {img_file}: {e}")
    
    if image_stats:
        widths = [s['width'] for s in image_stats]
        heights = [s['height'] for s in image_stats]
        sizes = [s['size'] for s in image_stats]
        
        print(f"📊 圖像質量統計 (基於 {len(image_stats)} 張圖像):")
        print(f"   平均尺寸: {np.mean(widths):.0f} x {np.mean(heights):.0f}")
        print(f"   尺寸範圍: {min(widths)}-{max(widths)} x {min(heights)}-{max(heights)}")
        print(f"   平均文件大小: {np.mean(sizes)/1024:.1f} KB")
        
        # 檢查尺寸一致性
        if len(set(widths)) > 1 or len(set(heights)) > 1:
            print("   ⚠️ 圖像尺寸不一致 - 可能影響訓練")
        else:
            print("   ✅ 圖像尺寸一致")
    else:
        print("❌ 沒有找到有效的圖像數據")
    
    return image_stats

def main():
    """主函數"""
    print("🔍 檢測性能問題調查 - 數據質量檢查")
    print("=" * 50)
    
    # 加載數據配置
    data_config = load_data_config()
    if data_config is None:
        return
    
    print(f"📁 數據集配置:")
    for split in ['train', 'val', 'test']:
        if split in data_config:
            print(f"   {split}: {data_config[split]}")
    
    # 檢查標註文件
    annotation_issues = check_annotation_files(data_config)
    
    # 分析類別分佈
    class_counts, total_objects, imbalance_ratio = analyze_class_distribution(data_config)
    
    # 分析目標尺寸
    sizes, areas, small_ratio = analyze_object_sizes(data_config)
    
    # 檢查圖像質量
    image_stats = check_image_quality(data_config)
    
    # 總結
    print("\n" + "=" * 50)
    print("📋 數據質量檢查總結:")
    
    if annotation_issues:
        print(f"   ❌ 發現 {len(annotation_issues)} 個標註問題")
    else:
        print("   ✅ 標註文件格式正確")
    
    if total_objects > 0:
        if imbalance_ratio > 3:
            print("   🚨 嚴重類別不平衡")
        elif imbalance_ratio > 2:
            print("   ⚠️ 中等類別不平衡")
        else:
            print("   ✅ 類別分佈相對平衡")
    else:
        print("   ❌ 沒有找到有效的標註數據")
    
    if areas is not None and len(areas) > 0:
        if small_ratio > 0.3:
            print("   🚨 小目標比例過高")
        elif small_ratio > 0.2:
            print("   ⚠️ 小目標比例較高")
        else:
            print("   ✅ 目標尺寸分佈正常")
    else:
        print("   ❌ 沒有找到有效的目標數據")
    
    print("\n🎯 建議下一步:")
    if annotation_issues:
        print("   1. 修復標註文件問題")
    if total_objects > 0 and imbalance_ratio > 2:
        print("   2. 考慮使用類別權重或平衡採樣")
    if areas is not None and len(areas) > 0 and small_ratio > 0.2:
        print("   3. 優化小目標檢測策略")
    print("   4. 繼續檢查模型架構問題")

if __name__ == "__main__":
    main()
