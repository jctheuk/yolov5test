#!/usr/bin/env python3
"""
分析標準 YOLOv5 偏置初始化是否會導致 NaN
Analyze if standard YOLOv5 bias initialization will cause NaN
"""

import math
import torch

def analyze_bias_initialization():
    """
    分析標準 YOLOv5 偏置初始化的數值穩定性
    """
    print("=== 標準 YOLOv5 偏置初始化分析 ===")
    
    # 典型的 stride 值
    strides = [8, 16, 32]  # P3, P4, P5
    nc = 4  # 類別數
    
    print("\n1. Objectness bias 計算:")
    for s in strides:
        obj_bias = math.log(8 / (640 / s) ** 2)
        print(f"   Stride {s}: log(8 / (640/{s})²) = log(8 / {640/s}²) = log(8 / {(640/s)**2}) = {obj_bias:.4f}")
    
    print("\n2. Classification bias 計算:")
    for s in strides:
        cls_bias = math.log(0.6 / (nc - 0.99999))
        print(f"   NC {nc}: log(0.6 / ({nc} - 0.99999)) = log(0.6 / {nc - 0.99999}) = {cls_bias:.4f}")
    
    print("\n3. 檢查是否會產生 NaN 或 inf:")
    for s in strides:
        obj_bias = math.log(8 / (640 / s) ** 2)
        cls_bias = math.log(0.6 / (nc - 0.99999))
        
        print(f"   Stride {s}:")
        print(f"     Objectness bias: {obj_bias} (isnan: {math.isnan(obj_bias)}, isinf: {math.isinf(obj_bias)})")
        print(f"     Classification bias: {cls_bias} (isnan: {math.isnan(cls_bias)}, isinf: {math.isinf(cls_bias)})")
    
    print("\n4. 檢查極端情況:")
    # 檢查如果 stride 為 0 或負數
    try:
        bad_bias = math.log(8 / (640 / 0) ** 2)
        print(f"   Stride 0: {bad_bias}")
    except:
        print("   Stride 0: 會產生除零錯誤")
    
    # 檢查如果 nc 為 1
    try:
        bad_cls = math.log(0.6 / (1 - 0.99999))
        print(f"   NC=1: {bad_cls}")
    except:
        print("   NC=1: 會產生除零錯誤")
    
    print("\n5. 數值範圍分析:")
    print("   Objectness bias 範圍:")
    for s in strides:
        obj_bias = math.log(8 / (640 / s) ** 2)
        print(f"     Stride {s}: {obj_bias:.4f}")
    
    print("   Classification bias:")
    cls_bias = math.log(0.6 / (nc - 0.99999))
    print(f"     NC {nc}: {cls_bias:.4f}")
    
    print("\n6. 結論:")
    print("   ✅ 標準 YOLOv5 偏置初始化是數值穩定的")
    print("   ✅ 不會產生 NaN 或 inf 值")
    print("   ✅ 所有偏置值都是有限的負數")
    print("   ✅ 只有在極端情況下（stride=0 或 nc=1）才會出現問題")
    print("   ✅ 在正常使用情況下是安全的")

def test_torch_implementation():
    """
    測試 PyTorch 實現的數值穩定性
    """
    print("\n=== PyTorch 實現測試 ===")
    
    # 創建測試張量
    device = torch.device('cpu')
    strides = torch.tensor([8, 16, 32], device=device)
    nc = 4
    
    print("\n1. PyTorch objectness bias 計算:")
    for s in strides:
        obj_bias = torch.log(torch.tensor(8.0) / (torch.tensor(640.0) / s) ** 2)
        print(f"   Stride {s.item()}: {obj_bias.item():.4f}")
    
    print("\n2. PyTorch classification bias 計算:")
    cls_bias = torch.log(torch.tensor(0.6) / (torch.tensor(nc) - 0.99999))
    print(f"   NC {nc}: {cls_bias.item():.4f}")
    
    print("\n3. 檢查 PyTorch 中的 NaN 和 inf:")
    for s in strides:
        obj_bias = torch.log(torch.tensor(8.0) / (torch.tensor(640.0) / s) ** 2)
        cls_bias = torch.log(torch.tensor(0.6) / (torch.tensor(nc) - 0.99999))
        
        print(f"   Stride {s.item()}:")
        print(f"     Objectness bias: {obj_bias.item():.4f} (isnan: {torch.isnan(obj_bias).item()}, isinf: {torch.isinf(obj_bias).item()})")
        print(f"     Classification bias: {cls_bias.item():.4f} (isnan: {torch.isnan(cls_bias).item()}, isinf: {torch.isinf(cls_bias).item()})")

if __name__ == "__main__":
    analyze_bias_initialization()
    test_torch_implementation()
