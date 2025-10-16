# YOLOv5MLC Models (Medium-Large Classification)

## 🎯 命名說明

**MLC = Medium-Large Classification**

這些模型介於 YOLOv5m (Medium) 和 YOLOv5l (Large) 之間，專為解決 99.7% GPU 記憶體使用率問題而設計。

## 📊 模型配置

### 核心參數
```yaml
depth_multiple: 0.83   # 介於 YOLOv5m(0.67) 和 YOLOv5l(1.0) 之間
width_multiple: 0.875  # 介於 YOLOv5m(0.75) 和 YOLOv5l(1.0) 之間
```

### 模型大小對比

| 模型類型 | depth | width | 參數量 | GPU記憶體(batch 64) |
|----------|-------|-------|--------|---------------------|
| YOLOv5m | 0.67 | 0.75 | ~21M | ~16GB |
| **YOLOv5MLC** | **0.83** | **0.875** | **~30M** | **~18-20GB** |
| YOLOv5l | 1.0 | 1.0 | ~47M | 31.9GB (99.7%!) |

## 📁 可用模型文件

### 4 個配置文件
1. **`yolov5mlc_classify_backbone.yaml`**
   - 分類頭連接到 backbone
   - 896 通道分類頭
   - 適合全局特徵分類

2. **`yolov5mlc_p3.yaml`**
   - 分類頭在 P3 層 (高解析度)
   - 224 通道分類頭
   - 適合小物體和細節特徵

3. **`yolov5mlc_p4.yaml`**
   - 分類頭在 P4 層 (中解析度)
   - 448 通道分類頭
   - 平衡性能和效率

4. **`yolov5mlc_p5.yaml`**
   - 分類頭在 P5 層 (低解析度)
   - 896 通道分類頭
   - 全局語義特徵

## 🚀 訓練命令

### 推薦配置 (批次 64 + 混合精度)
```bash
# Backbone 配置
python train.py \
  --cfg models/yolov5mlc_classify_backbone.yaml \
  --batch-size 64 \
  --imgsz 416 \
  --epochs 300 \
  --patience 0 \
  --amp \
  --cos-lr

# P3 配置 (可用更大批次)
python train.py \
  --cfg models/yolov5mlc_p3.yaml \
  --batch-size 128 \
  --imgsz 416 \
  --epochs 300 \
  --patience 0 \
  --amp \
  --cos-lr
```

### 完整訓練命令
查看 `yolov5mlc_training_commands.txt` 獲取所有 V1-V5 的完整命令。

## 📊 預期性能

### 記憶體使用
```
配置              GPU記憶體    使用率    安全性
─────────────────────────────────────────
原始(LC) + 128    31.9GB      99.7%     ❌ 極危險
MLC + 128         ~24.4GB     76%       ⚠️ 中等風險
MLC + 64          ~16-20GB    50-62%    ✅ 安全
MLC + 32          ~12-15GB    38-47%    ✅ 極安全
```

### 訓練性能
- **比 YOLOv5m 更好**: 約 +20-30% 參數量
- **比 YOLOv5l 更快**: 約 -35% 計算量
- **記憶體節省**: 約 -25~35% (相比 YOLOv5l)

### 模型精度預期
- **mAP**: 預期介於 YOLOv5m 和 YOLOv5l 之間
- **分類準確率**: 預期達到或接近 YOLOv5l 水準
- **訓練穩定性**: 顯著優於 YOLOv5l (無 NaN 錯誤)

## 💡 使用建議

### 何時使用 YOLOv5MLC？
1. ✅ GPU 記憶體有限 (32GB 卡)
2. ✅ 需要比 YOLOv5m 更好的性能
3. ✅ YOLOv5l 訓練失敗或不穩定
4. ✅ 需要平衡性能和記憶體
5. ✅ 生產環境部署考慮

### 何時使用原始 YOLOv5l？
1. ❌ 有充足 GPU 記憶體 (40GB+ 卡)
2. ❌ 追求極致精度
3. ❌ 可以使用極小批次大小 (16)
4. ❌ 訓練時間不重要

## 📋 文件結構

```
yolov5c/
├── models/
│   ├── yolov5mlc_classify_backbone.yaml
│   ├── yolov5mlc_p3.yaml
│   ├── yolov5mlc_p4.yaml
│   └── yolov5mlc_p5.yaml
│
├── yolov5mlc_training_commands.txt      # 訓練命令
├── YOLOV5MLC_MODELS_README.md          # 本文件
├── LIGHT_VS_ORIGINAL_ANALYSIS.md       # 詳細對比分析
├── MODEL_SIZE_COMPARISON.md            # 模型大小比較
└── LIGHTWEIGHT_MODEL_ANALYSIS.md       # 輕量化分析

## 🔍 技術細節

### 通道數計算
```python
# YOLOv5l (width=1.0)
P3: 256 channels
P4: 512 channels  
P5: 1024 channels

# YOLOv5MLC (width=0.875)
P3: 224 channels  # 256 × 0.875
P4: 448 channels  # 512 × 0.875
P5: 896 channels  # 1024 × 0.875
```

### 深度計算
```python
# C3 模塊重複次數
配置: [-1, 6, C3, [256]]

YOLOv5l:   round(6 × 1.0) = 6 個 C3
YOLOv5MLC: round(6 × 0.83) = 5 個 C3
YOLOv5m:   round(6 × 0.67) = 4 個 C3
```

### 記憶體縮放
```
通道縮放因子: 0.875
記憶體縮放因子: 0.875² = 0.766 (通道影響是平方關係)
總記憶體: 31.9GB × 0.766 = 24.4GB (batch 128)
```

## 🎓 命名規範

### 原始命名邏輯
- `yolov5lc` = YOLOv5 Large with Classification
- `_light` = 輕量化版本 (容易誤解為很小)

### 新命名邏輯  
- `yolov5mlc` = YOLOv5 Medium-Large with Classification
- 明確表示介於 Medium 和 Large 之間
- 更準確反映模型定位

## 📖 相關文檔

- `LIGHT_VS_ORIGINAL_ANALYSIS.md` - MLC vs 原始模型詳細對比
- `MODEL_SIZE_COMPARISON.md` - 各模型大小完整對比表
- `LIGHTWEIGHT_MODEL_ANALYSIS.md` - 輕量化策略分析
- `GPU_MEMORY_ANALYSIS.md` - GPU 記憶體使用分析
- `TRAINING_PROBLEM_SOLUTIONS.md` - 訓練問題解決方案

---

**總結**: YOLOv5MLC 提供了一個在性能和記憶體使用之間的最佳平衡點，特別適合在 32GB GPU 上進行醫學圖像的檢測和分類任務。

