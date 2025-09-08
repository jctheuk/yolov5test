#!/usr/bin/env python3
"""
測試修正後的偏置初始化
"""

import torch
from models.yolo import DetectionModel

def test_fixed_initialization():
    print("🔍 測試修正後的偏置初始化...")
    
    # 加載模型
    model_cfg = 'models/yolov5sc.yaml'
    model = DetectionModel(model_cfg, ch=3, nc=4)
    
    print("✅ 模型加載成功")
    print("📊 偏置初始化結果:")
    
    # 檢查偏置值
    for i, m in enumerate(model.model):
        if hasattr(m, 'm') and hasattr(m, 'stride'):  # Detect layer
            for j, mi in enumerate(m.m):
                if hasattr(mi, 'bias') and mi.bias is not None:
                    b = mi.bias.view(m.na, -1)
                    stride = m.stride[j] if hasattr(m.stride, '__getitem__') else m.stride
                    print(f"  層 {i}-{j} (stride={stride}):")
                    print(f"    Objectness bias: {b.data[:, 4].mean().item():.4f}")
                    print(f"    Classification bias: {b.data[:, 5:5+m.nc].mean().item():.4f}")
    
    print("\n✅ 修正後的初始化測試完成！")
    print("🎯 現在可以使用修正的配置進行訓練，無需特定權重文件")

if __name__ == '__main__':
    test_fixed_initialization()
