# YOLOv5WithClassification 模型大小比較

## 🎯 更新後的模型配置

你說得對！我之前創建的「輕量化」版本實際上就是 YOLOv5m 的大小。現在我已經更新為比 YOLOv5m 大但比 YOLOv5l 小的「中等加強」版本。

## 📊 YOLOv5 標準模型規格

| 模型類型 | depth_multiple | width_multiple | 參數估計 | 記憶體估計 | 性能 |
|----------|----------------|----------------|----------|------------|------|
| **YOLOv5n** | 0.33 | 0.25 | ~1.9M | ~4GB | 最小 |
| **YOLOv5s** | 0.33 | 0.50 | ~7.2M | ~8GB | 小 |
| **YOLOv5m** | 0.67 | 0.75 | ~21M | ~16GB | 中等 |
| **YOLOv5l** | 1.0 | 1.0 | ~47M | ~32GB | 大 |
| **YOLOv5x** | 1.33 | 1.25 | ~87M | ~50GB+ | 最大 |

## 🔧 我們的自定義版本

### 當前配置：中等加強 (Medium Plus)
```yaml
depth_multiple: 0.83   # 介於 YOLOv5m(0.67) 和 YOLOv5l(1.0) 之間
width_multiple: 0.875  # 介於 YOLOv5m(0.75) 和 YOLOv5l(1.0) 之間
```

### 性能定位
| 特性 | YOLOv5m | **我們的版本** | YOLOv5l | 原始(失敗) |
|------|---------|----------------|---------|------------|
| depth_multiple | 0.67 | **0.83** | 1.0 | 1.0 |
| width_multiple | 0.75 | **0.875** | 1.0 | 1.0 |
| 預估參數 | ~21M | **~30M** | ~47M | ~47M |
| 預估記憶體 | ~16GB | **~22-26GB** | ~32GB | 31.9GB |
| GPU 使用率 | ~50% | **~69-81%** | ~100% | 99.7% |
| 性能期待 | 中等 | **中等+** | 高 | 高(但失敗) |

## 🎯 為什麼選擇這個配置？

### 1. 比 YOLOv5m 更強
- **+24% depth**: 0.67 → 0.83 (更深的網路)
- **+17% width**: 0.75 → 0.875 (更寬的通道)
- **更好的特徵提取能力**

### 2. 比 YOLOv5l 更省記憶體
- **-17% depth**: 1.0 → 0.83 (減少層數)
- **-12.5% width**: 1.0 → 0.875 (減少通道)
- **大約節省 25-35% 記憶體**

### 3. 安全的記憶體使用率
- **預期使用**: 22-26GB (69-81% 使用率)
- **安全餘量**: 6-10GB 可用空間
- **避免 99.7% 危險狀況**

## 📁 更新的文件

### 模型配置文件
- `yolov5lc_classify_backbone_light.yaml` (896 通道)
- `yolov5lc_p3_light.yaml` (224 通道)
- `yolov5lc_p4_light.yaml` (448 通道)
- `yolov5lc_p5_light.yaml` (896 通道)

### 訓練命令
- `lightweight_model_training_commands.txt` (已更新)
- 批次大小: 64 (合理大小)
- 混合精度: `--amp` 啟用

## 🚀 推薦使用策略

### 優先順序 1: 中等加強模型
```bash
# 例如：Backbone V1
python train.py --cfg models/yolov5lc_classify_backbone_light.yaml --batch-size 64 --amp
```
- **優點**: 比 YOLOv5m 更強，記憶體安全
- **適用**: 大多數情況的最佳平衡

### 優先順序 2: 極致記憶體優化 (如果仍不足)
```bash
# 使用原始 YOLOv5l 但批次大小 16
python train.py --cfg models/yolov5lc_classify_backbone.yaml --batch-size 16 --amp
```
- **優點**: 最強性能，但訓練較慢
- **適用**: 記憶體仍不足時的備選

### 優先順序 3: YOLOv5m 基準 (如果需要更小)
如果你想創建真正的 YOLOv5m 版本用於比較：
```yaml
depth_multiple: 0.67   # YOLOv5m 標準
width_multiple: 0.75   # YOLOv5m 標準
```

## 💡 建議測試流程

1. **先測試中等加強版本** (當前配置)
2. **監控記憶體使用** (目標 <26GB)
3. **比較性能表現** (期待比 YOLOv5m 更好)
4. **如有問題再調整** (降批次大小或進一步縮小模型)

這個配置應該能給你比 YOLOv5m 更好的性能，同時避免原始 YOLOv5l 的記憶體問題！


