#!/usr/bin/env python3
"""
檢測性能問題調查 - 模型架構分析
分析 Backbone、檢測頭、Anchor 配置等
"""

import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def load_model_config():
    """加載模型配置"""
    try:
        with open('yolov5c/models/yolov5sc_classify_backbone.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        return model_config
    except Exception as e:
        print(f"❌ 無法加載模型配置: {e}")
        return None

def load_hyperparameters():
    """加載超參數配置"""
    try:
        with open('yolov5c/data/hyps/hyp.constraint_priority.yaml', 'r') as f:
            hyp_config = yaml.safe_load(f)
        return hyp_config
    except Exception as e:
        print(f"❌ 無法加載超參數配置: {e}")
        return None

def analyze_backbone_architecture(model_config):
    """分析 Backbone 架構"""
    print("🔍 分析 Backbone 架構...")
    
    backbone = model_config.get('backbone', [])
    print(f"📊 Backbone 層數: {len(backbone)}")
    
    # 分析通道數變化
    channels = []
    layer_types = []
    
    for i, layer in enumerate(backbone):
        if len(layer) >= 4:
            layer_type = layer[2] if len(layer) > 2 else "Unknown"
            layer_types.append(layer_type)
            
            # 提取通道數
            if layer_type == 'Conv' and len(layer[3]) > 0:
                channels.append(layer[3][0])
            elif layer_type == 'C3' and len(layer[3]) > 0:
                channels.append(layer[3][0])
            elif layer_type == 'SPPF' and len(layer[3]) > 0:
                channels.append(layer[3][0])
    
    print(f"📈 通道數變化: {channels}")
    print(f"🏗️ 層類型: {layer_types}")
    
    # 檢查醫學圖像適合性
    print(f"\n🏥 醫學圖像適合性分析:")
    
    # 檢查特徵提取深度
    conv_layers = [lt for lt in layer_types if lt == 'Conv']
    c3_layers = [lt for lt in layer_types if lt == 'C3']
    
    print(f"   Conv 層數: {len(conv_layers)}")
    print(f"   C3 層數: {len(c3_layers)}")
    print(f"   總層數: {len(backbone)}")
    
    if len(conv_layers) >= 4:
        print("   ✅ 特徵提取深度足夠")
    else:
        print("   ⚠️ 特徵提取深度可能不足")
    
    # 檢查通道數設計
    if len(channels) > 0:
        max_channels = max(channels)
        min_channels = min(channels)
        channel_ratio = max_channels / min_channels if min_channels > 0 else 0
        
        print(f"   最大通道數: {max_channels}")
        print(f"   最小通道數: {min_channels}")
        print(f"   通道比例: {channel_ratio:.2f}")
        
        if channel_ratio > 8:
            print("   ✅ 通道數設計合理，適合多尺度特徵提取")
        else:
            print("   ⚠️ 通道數設計可能過於保守")
    
    return channels, layer_types

def analyze_detection_head(model_config):
    """分析檢測頭設計"""
    print("\n🔍 分析檢測頭設計...")
    
    head = model_config.get('head', [])
    print(f"📊 檢測頭層數: {len(head)}")
    
    # 分析檢測頭結構
    detection_layers = []
    classification_layers = []
    
    for i, layer in enumerate(head):
        if len(layer) >= 3:
            layer_type = layer[2] if len(layer) > 2 else "Unknown"
            
            if layer_type == 'Detect':
                detection_layers.append((i, layer))
            elif layer_type == 'YOLOv5WithClassification':
                classification_layers.append((i, layer))
    
    print(f"🎯 檢測層數: {len(detection_layers)}")
    print(f"📋 分類層數: {len(classification_layers)}")
    
    # 檢查檢測頭配置
    for i, (layer_idx, layer) in enumerate(detection_layers):
        print(f"   檢測頭 {i+1}: {layer}")
        
        if len(layer) >= 4:
            args = layer[3]
            if len(args) >= 2:
                num_classes = args[0]
                anchors = args[1]
                print(f"     類別數: {num_classes}")
                print(f"     Anchor 數: {len(anchors) if isinstance(anchors, list) else 'N/A'}")
    
    # 檢查分類頭配置
    for i, (layer_idx, layer) in enumerate(classification_layers):
        print(f"   分類頭 {i+1}: {layer}")
        
        if len(layer) >= 4:
            args = layer[3]
            if len(args) >= 2:
                input_channels = args[0]
                num_classes = args[1]
                print(f"     輸入通道: {input_channels}")
                print(f"     分類類別數: {num_classes}")
    
    return detection_layers, classification_layers

def analyze_anchor_configuration(model_config):
    """分析 Anchor 配置"""
    print("\n🔍 分析 Anchor 配置...")
    
    anchors = model_config.get('anchors', [])
    print(f"📊 Anchor 配置: {anchors}")
    
    if anchors:
        # 分析 Anchor 尺寸
        anchor_sizes = []
        for i, anchor_set in enumerate(anchors):
            if isinstance(anchor_set, list) and len(anchor_set) >= 6:
                # 計算每個 anchor 的面積
                for j in range(0, len(anchor_set), 2):
                    if j + 1 < len(anchor_set):
                        w, h = anchor_set[j], anchor_set[j+1]
                        area = w * h
                        anchor_sizes.append(area)
                        print(f"   P{i+3}/8: Anchor {j//2+1}: {w}x{h} (面積: {area})")
        
        if anchor_sizes:
            min_area = min(anchor_sizes)
            max_area = max(anchor_sizes)
            area_ratio = max_area / min_area if min_area > 0 else 0
            
            print(f"\n📏 Anchor 面積分析:")
            print(f"   最小面積: {min_area}")
            print(f"   最大面積: {max_area}")
            print(f"   面積比例: {area_ratio:.2f}")
            
            # 檢查是否適合醫學圖像
            if min_area < 100:
                print("   ✅ 包含小目標 Anchor，適合醫學圖像")
            else:
                print("   ⚠️ 缺少小目標 Anchor，可能不適合醫學圖像")
            
            if area_ratio > 10:
                print("   ✅ Anchor 尺寸範圍足夠，適合多尺度目標")
            else:
                print("   ⚠️ Anchor 尺寸範圍可能不足")
    
    return anchors

def analyze_feature_fusion(model_config):
    """分析特徵融合策略"""
    print("\n🔍 分析特徵融合策略...")
    
    head = model_config.get('head', [])
    
    # 查找特徵融合層
    concat_layers = []
    upsample_layers = []
    
    for i, layer in enumerate(head):
        if len(layer) >= 3:
            layer_type = layer[2] if len(layer) > 2 else "Unknown"
            
            if layer_type == 'Concat':
                concat_layers.append((i, layer))
            elif layer_type == 'nn.Upsample':
                upsample_layers.append((i, layer))
    
    print(f"🔗 特徵融合層數: {len(concat_layers)}")
    print(f"⬆️ 上採樣層數: {len(upsample_layers)}")
    
    # 分析 FPN 結構
    print(f"\n🏗️ FPN 結構分析:")
    
    for i, (layer_idx, layer) in enumerate(concat_layers):
        print(f"   融合層 {i+1}: {layer}")
        
        if len(layer) >= 4:
            args = layer[3]
            if len(args) >= 1:
                concat_dim = args[0]
                print(f"     融合維度: {concat_dim}")
    
    # 檢查多尺度特徵融合
    if len(concat_layers) >= 3:
        print("   ✅ 多尺度特徵融合完整 (P3, P4, P5)")
    else:
        print("   ⚠️ 多尺度特徵融合可能不完整")
    
    return concat_layers, upsample_layers

def analyze_loss_weights(hyp_config):
    """分析損失權重配置"""
    print("\n🔍 分析損失權重配置...")
    
    if not hyp_config:
        print("❌ 無法加載超參數配置")
        return
    
    # 檢測損失權重
    box_weight = hyp_config.get('box', 0.05)
    cls_weight = hyp_config.get('cls', 0.5)
    obj_weight = hyp_config.get('obj', 1.0)
    
    print(f"📊 檢測損失權重:")
    print(f"   Box Loss: {box_weight}")
    print(f"   Class Loss: {cls_weight}")
    print(f"   Object Loss: {obj_weight}")
    
    # 分類任務權重
    cls_task_weight = hyp_config.get('cls_task', 0.1)
    constraint_weight = hyp_config.get('constraint_weight', 0.5)
    
    print(f"\n📋 分類任務權重:")
    print(f"   Classification Task: {cls_task_weight}")
    print(f"   Anatomical Constraint: {constraint_weight}")
    
    # 分析權重平衡
    total_detection_weight = box_weight + cls_weight + obj_weight
    total_classification_weight = cls_task_weight + constraint_weight
    
    print(f"\n⚖️ 權重平衡分析:")
    print(f"   檢測總權重: {total_detection_weight}")
    print(f"   分類總權重: {total_classification_weight}")
    
    if total_detection_weight > total_classification_weight * 2:
        print("   ⚠️ 檢測權重過高，可能影響分類性能")
    elif total_classification_weight > total_detection_weight * 2:
        print("   ⚠️ 分類權重過高，可能影響檢測性能")
    else:
        print("   ✅ 權重平衡相對合理")
    
    # 檢查 Box Loss 權重
    if box_weight < 0.1:
        print("   ⚠️ Box Loss 權重較低，可能影響邊界框回歸")
    else:
        print("   ✅ Box Loss 權重合理")
    
    return {
        'box': box_weight,
        'cls': cls_weight,
        'obj': obj_weight,
        'cls_task': cls_task_weight,
        'constraint': constraint_weight
    }

def analyze_medical_image_suitability(model_config, hyp_config):
    """分析醫學圖像適合性"""
    print("\n🔍 分析醫學圖像適合性...")
    
    # 檢查模型大小
    depth_multiple = model_config.get('depth_multiple', 0.33)
    width_multiple = model_config.get('width_multiple', 0.50)
    
    print(f"📏 模型大小配置:")
    print(f"   深度倍數: {depth_multiple}")
    print(f"   寬度倍數: {width_multiple}")
    
    # 分析是否適合醫學圖像
    if depth_multiple < 0.5 and width_multiple < 0.75:
        print("   ✅ 輕量級模型，適合醫學圖像的計算限制")
    else:
        print("   ⚠️ 模型較大，可能不適合醫學圖像的實時需求")
    
    # 檢查數據擴增設置
    if hyp_config:
        augmentation_params = {
            'hsv_h': hyp_config.get('hsv_h', 0),
            'hsv_s': hyp_config.get('hsv_s', 0),
            'hsv_v': hyp_config.get('hsv_v', 0),
            'degrees': hyp_config.get('degrees', 0),
            'translate': hyp_config.get('translate', 0),
            'scale': hyp_config.get('scale', 0)
        }
        
        print(f"\n🔄 數據擴增設置:")
        for param, value in augmentation_params.items():
            print(f"   {param}: {value}")
        
        # 檢查是否適合醫學圖像
        total_augmentation = sum(augmentation_params.values())
        if total_augmentation == 0:
            print("   ✅ 數據擴增已關閉，適合醫學圖像的準確性要求")
        else:
            print("   ⚠️ 數據擴增已啟用，可能影響醫學圖像的診斷準確性")
    
    return depth_multiple, width_multiple

def main():
    """主函數"""
    print("🔍 檢測性能問題調查 - 模型架構分析")
    print("=" * 50)
    
    # 加載配置
    model_config = load_model_config()
    hyp_config = load_hyperparameters()
    
    if model_config is None:
        return
    
    print(f"📁 模型配置:")
    print(f"   檢測類別數: {model_config.get('nc', 'N/A')}")
    print(f"   分類類別數: {model_config.get('num_cls', 'N/A')}")
    print(f"   深度倍數: {model_config.get('depth_multiple', 'N/A')}")
    print(f"   寬度倍數: {model_config.get('width_multiple', 'N/A')}")
    
    # 分析各個組件
    channels, layer_types = analyze_backbone_architecture(model_config)
    detection_layers, classification_layers = analyze_detection_head(model_config)
    anchors = analyze_anchor_configuration(model_config)
    concat_layers, upsample_layers = analyze_feature_fusion(model_config)
    loss_weights = analyze_loss_weights(hyp_config)
    depth_multiple, width_multiple = analyze_medical_image_suitability(model_config, hyp_config)
    
    # 總結
    print("\n" + "=" * 50)
    print("📋 模型架構分析總結:")
    
    # Backbone 分析
    if len(layer_types) >= 8:
        print("   ✅ Backbone 深度足夠")
    else:
        print("   ⚠️ Backbone 深度可能不足")
    
    # 檢測頭分析
    if len(detection_layers) >= 1:
        print("   ✅ 檢測頭配置完整")
    else:
        print("   ❌ 檢測頭配置有問題")
    
    # 分類頭分析
    if len(classification_layers) >= 1:
        print("   ✅ 分類頭配置完整")
    else:
        print("   ❌ 分類頭配置有問題")
    
    # Anchor 分析
    if anchors and len(anchors) >= 3:
        print("   ✅ Anchor 配置完整")
    else:
        print("   ⚠️ Anchor 配置可能不完整")
    
    # 特徵融合分析
    if len(concat_layers) >= 3:
        print("   ✅ 多尺度特徵融合完整")
    else:
        print("   ⚠️ 多尺度特徵融合可能不完整")
    
    # 權重平衡分析
    if loss_weights:
        total_detection = loss_weights['box'] + loss_weights['cls'] + loss_weights['obj']
        total_classification = loss_weights['cls_task'] + loss_weights['constraint']
        
        if abs(total_detection - total_classification) < total_detection * 0.5:
            print("   ✅ 損失權重平衡合理")
        else:
            print("   ⚠️ 損失權重可能不平衡")
    
    print("\n🎯 建議下一步:")
    print("   1. 檢查訓練策略問題")
    print("   2. 分析任務衝突問題")
    print("   3. 驗證技術實現問題")

if __name__ == "__main__":
    main()

