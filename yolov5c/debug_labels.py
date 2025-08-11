#!/usr/bin/env python3
"""
Debug script to check detection coordinates and classification labels
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the yolov5c directory to the path
sys.path.append(str(Path(__file__).parent))

from utils.dataloaders import create_dataloader
from utils.general import check_dataset

def debug_labels():
    print("=== 目標辨識資料集調試 ===")
    
    # Load dataset configuration
    data_yaml = "../Regurgitation-YOLODataset-Detection/data.yaml"
    data_dict = check_dataset(data_yaml)
    
    print(f"Dataset config: {data_dict}")
    
    # Create dataloader
    train_loader, dataset = create_dataloader(
        path=data_dict['train'],
        imgsz=416,
        batch_size=2,
        stride=32,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        workers=0,
        image_weights=False,
        quad=False,
        prefix='',
        shuffle=False,
        seed=0
    )
    
    print(f"\n=== 資料集統計 ===")
    print(f"總圖片數: {len(dataset.im_files)}")
    print(f"總批次數: {len(train_loader)}")
    print(f"檢測類別數: {data_dict['nc']}")
    print(f"分類類別數: {data_dict['num_cls']}")
    print(f"檢測類別名稱: {data_dict['names']}")
    print(f"分類類別名稱: {data_dict['cls_names']}")
    
    # Check first few samples
    print(f"\n=== 前5個樣本詳細檢查 ===")
    for i in range(min(5, len(dataset.im_files))):
        print(f"\n--- 樣本 {i} ---")
        print(f"圖片路徑: {dataset.im_files[i]}")
        print(f"標籤路徑: {dataset.label_files[i]}")
        
        # Check detection labels
        detection_labels = dataset.labels[i]
        print(f"檢測標籤數量: {len(detection_labels)}")
        if len(detection_labels) > 0:
            print(f"檢測標籤形狀: {detection_labels.shape}")
            print(f"檢測標籤內容:")
            for j, label in enumerate(detection_labels):
                class_id = int(label[0])
                x, y, w, h = label[1:5]
                print(f"  標籤 {j}: 類別={class_id}({data_dict['names'][class_id]}), x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
        
        # Check classification labels
        classification_label = dataset.classification_labels[i]
        print(f"分類標籤原始值: {classification_label}")
        if classification_label is not None:
            if isinstance(classification_label, (list, tuple)):
                print(f"分類標籤類型: list/tuple, 長度: {len(classification_label)}")
                print(f"分類標籤值: {classification_label}")
            else:
                print(f"分類標籤類型: {type(classification_label)}, 值: {classification_label}")
        else:
            print("分類標籤: None")
    
    # Check batch processing
    print(f"\n=== 批次處理檢查 ===")
    for batch_idx, (imgs, targets, classification_labels, paths, shapes) in enumerate(train_loader):
        print(f"\n--- 批次 {batch_idx} ---")
        print(f"圖片形狀: {imgs.shape}")
        print(f"檢測目標形狀: {targets.shape}")
        print(f"分類標籤形狀: {classification_labels.shape}")
        print(f"圖片路徑: {paths}")
        
        # Check detection targets in batch
        if targets.numel() > 0:
            print(f"檢測目標內容:")
            for i in range(targets.shape[0]):
                target = targets[i]
                if target.numel() > 0:
                    print(f"  樣本 {i}: {target}")
                    # Check if coordinates are normalized (should be between 0 and 1)
                    if target.dim() == 1 and target.shape[0] >= 6:
                        coords = target[2:6]  # x, y, w, h
                        min_coords = coords.min().item()
                        max_coords = coords.max().item()
                        print(f"    座標範圍: [{min_coords:.4f}, {max_coords:.4f}]")
                        if min_coords < 0 or max_coords > 1:
                            print(f"    ⚠️ 警告: 座標超出正常範圍 [0, 1]")
                    elif target.dim() == 2:
                        coords = target[:, 2:6]  # x, y, w, h
                        if coords.numel() > 0:
                            min_coords = coords.min().item()
                            max_coords = coords.max().item()
                            print(f"    座標範圍: [{min_coords:.4f}, {max_coords:.4f}]")
                            if min_coords < 0 or max_coords > 1:
                                print(f"    ⚠️ 警告: 座標超出正常範圍 [0, 1]")
        
        # Check classification labels in batch
        print(f"分類標籤內容:")
        for i in range(classification_labels.shape[0]):
            cls_label = classification_labels[i]
            print(f"  樣本 {i}: {cls_label}")
            # Check if it's one-hot encoded
            if cls_label.sum() == 1.0:
                class_idx = cls_label.argmax().item()
                print(f"    分類類別: {class_idx} ({data_dict['cls_names'][class_idx]})")
            else:
                print(f"    ⚠️ 警告: 不是有效的one-hot編碼")
        
        # Only check first 2 batches
        if batch_idx >= 1:
            break
    
    # Check cache file
    print(f"\n=== 快取檔案檢查 ===")
    cache_path = Path(data_dict['train']).parent / 'labels.cache'
    if cache_path.exists():
        print(f"快取檔案存在: {cache_path}")
        try:
            cache_data = np.load(cache_path, allow_pickle=True).item()
            print(f"快取版本: {cache_data.get('version', 'unknown')}")
            print(f"快取雜湊: {cache_data.get('hash', 'unknown')}")
            print(f"快取結果: {cache_data.get('results', 'unknown')}")
            
            # Check cache structure
            cache_keys = list(cache_data.keys())
            cache_keys = [k for k in cache_keys if k not in ['version', 'hash', 'results', 'msgs']]
            if cache_keys:
                first_key = cache_keys[0]
                first_value = cache_data[first_key]
                print(f"快取結構範例 ({first_key}):")
                print(f"  檢測標籤: {first_value[0].shape if hasattr(first_value[0], 'shape') else type(first_value[0])}")
                print(f"  圖片形狀: {first_value[1]}")
                print(f"  分割標籤: {type(first_value[2])}")
                print(f"  分類標籤: {first_value[3] if len(first_value) > 3 else 'N/A'}")
        except Exception as e:
            print(f"讀取快取檔案時發生錯誤: {e}")
    else:
        print(f"快取檔案不存在: {cache_path}")
    
    print(f"\n=== 調試完成 ===")

if __name__ == "__main__":
    debug_labels() 