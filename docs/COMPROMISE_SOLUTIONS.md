# 折衷解決方案分析

## 🎯 您的需求

1. **保持可比較性** - batch size 不能離 128 太遠
2. **提升穩定性** - 降低梯度規模避免 NaN

---

## 💡 方案 A: 輕微降低 Batch Size

### **建議：Batch 96 或 Batch 112**

#### **Batch 112 (降低 12.5%)**

| 架構 | 梯度張量大小 | vs Batch 128 | 預期效果 |
|------|-------------|-------------|---------|
| P3 Large | 52×52 × 112 × 1024 = 310M | -12.5% | 可能不夠 |
| P4 Large | 26×26 × 112 × 1280 = 97M | -12.5% | 應該有幫助 |
| P5 Large | 13×13 × 112 × 1280 = 24M | -12.5% | 應該有幫助 |

**優點：**
- ✅ 接近原始配置（112 vs 128）
- ✅ 結果可比較性高
- ✅ 訓練時間幾乎不變

**缺點：**
- ⚠️ 降幅可能不夠（只減 12.5%）
- ⚠️ P3 可能仍然不穩定

---

#### **Batch 96 (降低 25%)** ⭐ 推薦

| 架構 | 梯度張量大小 | vs Batch 128 | 預期效果 |
|------|-------------|-------------|---------|
| P3 Large | 52×52 × 96 × 1024 = 266M | **-25%** | 應該穩定 |
| P4 Large | 26×26 × 96 × 1280 = 83M | -25% | 應該很穩定 |
| P5 Large | 13×13 × 96 × 1280 = 21M | -25% | 應該很穩定 |

**優點：**
- ✅ 降幅適中（25%）
- ✅ 仍然接近 128（可比較）
- ✅ P3 梯度從 354M 降到 266M (接近 Medium 的 266M)
- ✅ 訓練時間增加約 15%

**缺點：**
- 🔶 不是 2 的冪次（但不影響功能）

---

## 💡 方案 B: 介於 Large 和 Medium 之間的模型

### **建議：Width 0.875, Depth 0.835**

**計算：**
```
Width:  (0.75 + 1.0) / 2 = 0.875
Depth:  (0.67 + 1.0) / 2 = 0.835
```

#### **實際通道數對比：**

| 層 | Medium (0.75) | **Compromise (0.875)** | Large (1.0) |
|----|--------------|----------------------|------------|
| P3 | 192 | **224** | 256 |
| P4 | 384 | **448** | 512 |
| P5 | 768 | **896** | 1024 |

#### **分類頭中間通道：**

| 架構 | Medium | **Compromise** | Large |
|------|--------|---------------|-------|
| P3 | 768 | **896** | 1024 |
| P4 | 1280 (限制) | **1280** (限制) | 1280 (限制) |
| P5 | 1280 (限制) | **1280** (限制) | 1280 (限制) |

#### **梯度張量（Batch 128）：**

```
P3 Compromise: 52×52 × 128 × 896 = 310M elements
  vs Large: 354M (-12.5%)
  vs Medium: 266M (+16.5%)

接近 Medium，比 Large 小 12.5%
```

**優點：**
- ✅ Batch 保持 128（完全可比較）
- ✅ 性能接近 Large
- ✅ P3 梯度降低 12.5%

**缺點：**
- ⚠️ 需要創建新的模型配置文件
- ⚠️ 降幅可能不夠（只減 12.5%）
- ⚠️ P4/P5 沒變化（都被 1280 限制）

---

## 🎯 推薦方案組合

### **最佳折衷：Batch 96 + Large 模型** ⭐

```bash
--batch-size 96 --cfg models/yolov5lc_p3.yaml
```

**為什麼：**
1. ✅ **可比較性高** - 96 接近 128 (只差 25%)
2. ✅ **穩定性好** - 梯度降低 25%
3. ✅ **不需改模型** - 使用現有 Large 模型
4. ✅ **訓練時間增加少** - 約 15%

**梯度效果：**
```
P3 Large batch 96: 266M elements
  = P3 Medium batch 128 的大小 ✅
```

---

## 📊 方案對比表

| 方案 | Batch | Width | Depth | P3 梯度 | 可比性 | 穩定性 | 推薦度 |
|------|-------|-------|-------|---------|--------|--------|-------|
| 原配置 | 128 | 1.0 | 1.0 | 354M | 100% | 低 ❌ | - |
| **Batch 96** | **96** | **1.0** | **1.0** | **266M** | **75%** | **高** | ⭐⭐⭐⭐⭐ |
| Batch 112 | 112 | 1.0 | 1.0 | 310M | 87% | 中 | ⭐⭐⭐ |
| Batch 64 | 64 | 1.0 | 1.0 | 177M | 50% | 很高 | ⭐⭐⭐⭐ |
| Compromise 模型 | 128 | 0.875 | 0.835 | 310M | 100% | 中 | ⭐⭐⭐ |
| Medium 模型 | 128 | 0.75 | 0.67 | 266M | 100% | 高 | ⭐⭐⭐⭐ |

---

## 🎯 具體執行建議

### **階段 1: 快速測試（推薦三個配置）**

#### **Test 1: Batch 96 (最推薦)** ⭐
```bash
cd yolov5c

python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5lc_p4.yaml \
    --hyp data/hyps/hyp.default.yaml \
    --epochs 50 \
    --batch-size 96 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name test_v2_batch96
```

#### **Test 2: Batch 112 (最接近原始)**
```bash
python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5lc_p4.yaml \
    --hyp data/hyps/hyp.default.yaml \
    --epochs 50 \
    --batch-size 112 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name test_v2_batch112
```

#### **Test 3: Medium 模型 (作為對照)**
```bash
python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5mc_p4.yaml \
    --hyp data/hyps/hyp.default.yaml \
    --epochs 50 \
    --batch-size 128 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name test_v2_medium
```

---

### **階段 2: 根據測試結果選擇**

| 測試結果 | 選擇方案 | 理由 |
|---------|---------|------|
| Batch 112 成功 | 用 Batch 112 | 最接近原始，可比性最高 |
| Batch 96 成功，112 失敗 | 用 Batch 96 | 平衡可比性和穩定性 ⭐ |
| 都失敗 | 用 Batch 64 或 Medium 模型 | 確保成功 |

---

## 📋 重訓命令（Batch 96 版本）

### **P3 Model - 4 tasks**
```bash
python train.py --data ../regurgitationV1/data.yaml --cfg models/yolov5lc_p3.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p3_v1

python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p3.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p3_v2

python train.py --data ../regurgitationV3/data.yaml --cfg models/yolov5lc_p3.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p3_v3

python train.py --data ../regurgitationV5/data.yaml --cfg models/yolov5lc_p3.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p3_v5
```

### **P4 Model - 2 tasks**
```bash
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p4.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p4_v2

python train.py --data ../regurgitationV3/data.yaml --cfg models/yolov5lc_p4.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p4_v3
```

### **P5 Model - 3 tasks**
```bash
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p5.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p5_v2

python train.py --data ../regurgitationV4/data.yaml --cfg models/yolov5lc_p5.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p5_v4

python train.py --data ../regurgitationV5/data.yaml --cfg models/yolov5lc_p5.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_p5_v5
```

### **Classify Backbone Model - 3 tasks**
```bash
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_classify_backbone.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_backbone_v2

python train.py --data ../regurgitationV4/data.yaml --cfg models/yolov5lc_classify_backbone.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_backbone_v4

python train.py --data ../regurgitationV5/data.yaml --cfg models/yolov5lc_classify_backbone.yaml --hyp data/hyps/hyp.default.yaml --epochs 300 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --nosave --project runs/train --name yolov5lc_backbone_v5
```

---

## 💡 方案 B: 介於 Large 和 Medium 的模型大小

### **建議：Width 0.875, Depth 0.85** ⭐

**計算接近 Large 的配置：**
```
Width:  Large(1.0) - 0.125 = 0.875
Depth:  Large(1.0) - 0.15 = 0.85
```

#### **通道數對比：**

| 層 | Medium (0.75) | **0.875** | Large (1.0) | 相對 Large |
|----|--------------|-----------|------------|-----------|
| P3 | 192 ch | **224 ch** | 256 ch | -12.5% |
| P4 | 384 ch | **448 ch** | 512 ch | -12.5% |
| P5 | 768 ch | **896 ch** | 1024 ch | -12.5% |

#### **分類頭中間通道：**

```python
c_ = min(1280, max(256, in_channels * 4))

P3: min(1280, 224 × 4) = min(1280, 896) = 896
    vs Large: 1024 (-12.5%)
    vs Medium: 768 (+16.7%)

P4: min(1280, 448 × 4) = min(1280, 1792) = 1280 (被限制)
    = Large = Medium (相同)

P5: min(1280, 896 × 4) = min(1280, 3584) = 1280 (被限制)
    = Large = Medium (相同)
```

#### **梯度張量大小（Batch 128）：**

```
P3 (0.875): 52×52 × 128 × 896 = 310M elements
  vs Large (1.0): 354M (-12.5%)
  vs Medium (0.75): 266M (+16.5%)
```

---

## 📊 兩個方案的詳細對比

| 項目 | 方案 A: Batch 96 | 方案 B: Width 0.875 | 原配置 |
|------|-----------------|-------------------|--------|
| **Batch Size** | **96** | 128 | 128 |
| **Width** | 1.0 | **0.875** | 1.0 |
| **Depth** | 1.0 | **0.85** | 1.0 |
| **P3 梯度** | 266M | 310M | 354M |
| **梯度降低** | **-25%** | -12.5% | 0% |
| **可比性** | 75% | 100% | 100% |
| **訓練時間** | +15% | 相同 | 基準 |
| **需修改** | 只改 batch | 需創建新 YAML | - |
| **推薦度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | - |

---

## 🔧 如何實施方案 B

### **創建新模型配置文件：**

```bash
# 複製 Large 模型並修改
cp yolov5c/models/yolov5lc_p3.yaml yolov5c/models/yolov5lc_p3_0875.yaml
```

**修改內容：**
```yaml
depth_multiple: 0.85  # 從 1.0 改為 0.85
width_multiple: 0.875  # 從 1.0 改為 0.875
```

**使用方式：**
```bash
python train.py \
    --cfg models/yolov5lc_p3_0875.yaml \
    --batch-size 128 \
    ...
```

---

## 🎯 執行計劃建議

### **Step 1: 快速診斷（選最接近原始的）**

```bash
# Test Batch 112 (只降 12.5%，最接近 128)
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p4.yaml --hyp data/hyps/hyp.default.yaml --epochs 50 --batch-size 112 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --project runs/train --name test_v2_batch112
```

**判斷：**
- ✅ 成功 (epoch 10+) → 用 Batch 112 重訓所有任務
- ❌ 失敗 (epoch <6) → 試 Batch 96

### **Step 2: 如果 Batch 112 失敗，試 Batch 96**

```bash
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p4.yaml --hyp data/hyps/hyp.default.yaml --epochs 50 --batch-size 96 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --project runs/train --name test_v2_batch96
```

### **Step 3: 根據成功的配置執行所有 12 個任務**

---

## 📈 預期結果

| 配置 | 成功機率 | 可比性 | 訓練時間 |
|------|---------|--------|---------|
| Batch 112 | 70% | 87% | +7% |
| **Batch 96** | **85%** | **75%** | **+15%** ⭐ |
| Batch 64 | 95% | 50% | +30% |

---

## ✅ 最終建議

**優先試 Batch 96**
- 在可比性 (75%) 和穩定性 (85%) 之間取得最佳平衡
- 不需要創建新模型文件
- 梯度降低 25% (足夠穩定)
- 接近原始配置 (論文中可以說明是為了穩定性的輕微調整)



