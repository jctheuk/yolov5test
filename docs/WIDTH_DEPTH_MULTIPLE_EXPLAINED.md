# Width Multiple 和 Depth Multiple 詳解

## 📍 代碼位置

**`yolov5c/models/yolo.py`** - `parse_model` 函數 (line 522, 543, 562)

---

## 🎯 什麼是 Width Multiple 和 Depth Multiple？

這是 YOLOv5 用來縮放模型大小的兩個參數：

### **1. Width Multiple (寬度倍數)**
**控制：每一層的通道數（channel數）**

```python
# yolov5c/models/yolo.py, line 562
c2 = make_divisible(c2 * gw, 8)  # gw = width_multiple
```

### **2. Depth Multiple (深度倍數)**
**控制：重複模塊的數量（層數）**

```python
# yolov5c/models/yolo.py, line 543
n = max(round(n * gd), 1) if n > 1 else n  # gd = depth_multiple
```

---

## 📊 YOLOv5 模型尺寸對比

| 模型 | Width | Depth | 參數量 | 用途 |
|------|-------|-------|--------|------|
| **YOLOv5n** (nano) | 0.25 | 0.33 | 1.9M | 最小、最快 |
| **YOLOv5s** (small) | 0.50 | 0.33 | 7.2M | 小、快速 |
| **YOLOv5m** (medium) | 0.75 | 0.67 | 21.2M | 平衡 |
| **YOLOv5l** (large) | **1.0** | **1.0** | **46.5M** | 大、準確 |
| **YOLOv5x** (xlarge) | 1.25 | 1.33 | 86.7M | 最大、最準 |

**您的訓練使用：Large (width=1.0, depth=1.0)**

---

## 🔧 Width Multiple 如何工作

### **範例：Conv 層 `[64, 6, 2, 2]`**

YAML 配置：
```yaml
[-1, 1, Conv, [64, 6, 2, 2]]
             ^^
          輸出通道數
```

**實際通道數 = 64 × width_multiple：**

| Width Multiple | 實際輸出通道 | 計算 |
|---------------|------------|------|
| 0.25 (nano) | 16 | 64 × 0.25 = 16 |
| 0.50 (small) | 32 | 64 × 0.50 = 32 |
| 0.75 (medium) | 48 | 64 × 0.75 = 48 |
| **1.0 (large)** | **64** | 64 × 1.0 = 64 |
| 1.25 (xlarge) | 80 | 64 × 1.25 = 80 |

### **應用到整個網絡：**

```yaml
backbone:
  Conv, [64]    → Small: 32,  Large: 64
  Conv, [128]   → Small: 64,  Large: 128
  Conv, [256]   → Small: 128, Large: 256
  Conv, [512]   → Small: 256, Large: 512
  Conv, [1024]  → Small: 512, Large: 1024
```

---

## 🔧 Depth Multiple 如何工作

### **範例：C3 模塊 `[3, C3, [256]]`**

YAML 配置：
```yaml
[-1, 3, C3, [256]]
     ^
  重複次數
```

**實際重複次數 = round(3 × depth_multiple)：**

| Depth Multiple | 實際重複次數 | 計算 |
|---------------|------------|------|
| 0.33 (small) | 1 | round(3 × 0.33) = 1 |
| 0.67 (medium) | 2 | round(3 × 0.67) = 2 |
| **1.0 (large)** | **3** | round(3 × 1.0) = 3 |
| 1.33 (xlarge) | 4 | round(3 × 1.33) = 4 |

### **應用到整個網絡：**

```yaml
backbone:
  [-1, 3, C3, [128]]   → Small: 1 個 C3,  Large: 3 個 C3
  [-1, 6, C3, [256]]   → Small: 2 個 C3,  Large: 6 個 C3
  [-1, 9, C3, [512]]   → Small: 3 個 C3,  Large: 9 個 C3
```

---

## 📊 實際影響範例

### **Backbone Layer 4: `[-1, 6, C3, [256]]`**

| 模型 | Width | Depth | 通道數 | C3 重複 | 總層數 |
|------|-------|-------|--------|--------|-------|
| Small | 0.50 | 0.33 | 128 | 2 | 2 個 C3×128ch |
| Medium | 0.75 | 0.67 | 192 | 4 | 4 個 C3×192ch |
| **Large** | **1.0** | **1.0** | **256** | **6** | **6 個 C3×256ch** |

**Large 比 Small：**
- 通道數 2 倍 (256 vs 128)
- 層數 3 倍 (6 vs 2)
- **參數量約 6 倍！**

---

## 🎯 對您的訓練的影響

### **YOLOv5lc Large (width=1.0, depth=1.0)**

**通道數：**
```
P3 輸入: 256 channels
P4 輸入: 512 channels
P5 輸入: 1024 channels
```

**分類頭中間層：**
```python
# common.py line 971
c_ = min(1280, max(256, in_channels * 4))

P3: min(1280, 256 × 4) = min(1280, 1024) = 1024 ✅ 沒被限制
P4: min(1280, 512 × 4) = min(1280, 2048) = 1280 ⚠️ 被限制
P5: min(1280, 1024 × 4) = min(1280, 4096) = 1280 ⚠️ 被限制
```

### **如果用 Small (width=0.5, depth=0.33)：**

**通道數：**
```
P3 輸入: 128 channels
P4 輸入: 256 channels
P5 輸入: 512 channels
```

**分類頭中間層：**
```
P3: min(1280, 128 × 4) = min(1280, 512) = 512 ✅ 減半！
P4: min(1280, 256 × 4) = min(1280, 1024) = 1024
P5: min(1280, 512 × 4) = min(1280, 2048) = 1280
```

**網絡深度：**
```
C3 重複：Small 只有 Large 的 1/3 層數
總參數：~7M (Large 是 ~47M)
```

---

## 💥 為什麼 Large 失敗，Small/Medium 成功？

### **梯度張量大小對比（Batch 128）：**

```
P3 分類頭中間層：

Small:  52×52 × 128 × 512  = 177M elements
Medium: 52×52 × 128 × 768  = 266M elements  
Large:  52×52 × 128 × 1024 = 354M elements  🔴

Large 是 Small 的 2 倍！
```

**加上 V2 的困難特徵分佈 → 梯度爆炸 → NaN 錯誤**

---

## 🔧 解決方案

### **選項 1: 降低 Batch Size (保持 Large 模型)**
```bash
--batch-size 64  # 從 128 降到 64

效果：Large batch 64 = Small batch 128 的梯度規模
```

### **選項 2: 使用 Small 模型 (保持 Batch 128)**
```bash
--cfg models/yolov5sc_p3.yaml  # width=0.5, depth=0.33

優點：更穩定、更快、GPU 記憶體少
缺點：準確率可能略低
```

### **選項 3: 使用 Medium 模型**
```bash
--cfg models/yolov5mc_p3.yaml  # width=0.75, depth=0.67

優點：平衡性能和穩定性
```

---

## 📋 總結

### **Width Multiple (寬度倍數):**
- **作用**: 縮放每層的通道數
- **代碼**: `channels × width_multiple`
- **影響**: 模型容量、記憶體、梯度規模

### **Depth Multiple (深度倍數):**
- **作用**: 縮放重複模塊的數量
- **代碼**: `round(repeats × depth_multiple)`
- **影響**: 模型深度、計算量、參數量

### **您的情況：**
- ✅ 使用 Large (width=1.0, depth=1.0)
- ✅ P3 的中間通道變成 1024（沒被 1280 限制）
- ✅ 梯度張量 354M elements
- ✅ 配合 batch 128 和 V2 數據 → NaN 錯誤
- ✅ **解決方案：batch 64**






