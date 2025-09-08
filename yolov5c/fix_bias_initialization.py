#!/usr/bin/env python3
"""
修正 YOLOv5WithClassification 的偏置初始化
基於診斷結果的修正版本
"""

import torch
import torch.nn as nn
import math
from models.yolo import DetectionModel

def fix_bias_initialization(model):
    """
    修正偏置初始化，解決 objectness loss 過小的問題
    """
    print("🔧 修正偏置初始化...")
    
    # 找到 Detect 層
    detect_layer = None
    for m in model.model:
        if hasattr(m, 'm') and hasattr(m, 'stride'):  # Detect layer
            detect_layer = m
            break
    
    if detect_layer is None:
        print("❌ 未找到 Detect 層")
        return False
    
    print(f"✅ 找到 Detect 層，stride: {detect_layer.stride}")
    
    # 修正每個卷積層的偏置
    for i, mi in enumerate(detect_layer.m):
        if hasattr(mi, 'bias') and mi.bias is not None:
            b = mi.bias.view(detect_layer.na, -1)
            stride = detect_layer.stride[i] if hasattr(detect_layer.stride, '__getitem__') else detect_layer.stride
            
            print(f"  修正層 {i} (stride={stride}):")
            
            # 保存原始值
            original_obj_bias = b.data[:, 4].clone()
            original_cls_bias = b.data[:, 5:5+detect_layer.nc].clone()
            
            # 修正 objectness 偏置 - 使用更保守的初始化
            # 原始公式: math.log(8 / (640 / s) ** 2)
            # 修正公式: math.log(0.5 / (640 / s) ** 2) - 更小的初始值
            new_obj_bias = math.log(0.5 / (640 / stride) ** 2)
            b.data[:, 4] = new_obj_bias
            
            # 修正分類偏置 - 使用更平衡的初始化
            new_cls_bias = math.log(0.8 / (detect_layer.nc - 0.99999))  # 從 0.6 改為 0.8
            b.data[:, 5:5+detect_layer.nc] = new_cls_bias
            
            # 更新參數
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            
            print(f"    Objectness bias: {original_obj_bias.mean().item():.4f} -> {new_obj_bias:.4f}")
            print(f"    Classification bias: {original_cls_bias.mean().item():.4f} -> {new_cls_bias:.4f}")
    
    print("✅ 偏置初始化修正完成")
    return True

def main():
    """主函數"""
    print("🔍 開始修正偏置初始化...")
    
    # 加載模型
    model_cfg = 'models/yolov5sc.yaml'
    model = DetectionModel(model_cfg, ch=3, nc=4)
    
    print("✅ 模型加載成功")
    
    # 修正偏置初始化
    success = fix_bias_initialization(model)
    
    if success:
        # 保存修正後的模型
        torch.save(model.state_dict(), 'yolov5sc_fixed_bias.pt')
        print("✅ 修正後的模型已保存為 yolov5sc_fixed_bias.pt")
        
        # 驗證修正結果
        print("\n📊 驗證修正結果:")
        for i, m in enumerate(model.model):
            if hasattr(m, 'm') and hasattr(m, 'stride'):
                for j, mi in enumerate(m.m):
                    if hasattr(mi, 'bias') and mi.bias is not None:
                        b = mi.bias.view(m.na, -1)
                        stride = m.stride[j] if hasattr(m.stride, '__getitem__') else m.stride
                        print(f"  層 {i}-{j} (stride={stride}):")
                        print(f"    Objectness bias: {b.data[:, 4].mean().item():.4f}")
                        print(f"    Classification bias: {b.data[:, 5:5+m.nc].mean().item():.4f}")
    else:
        print("❌ 偏置初始化修正失敗")

if __name__ == '__main__':
    main()
