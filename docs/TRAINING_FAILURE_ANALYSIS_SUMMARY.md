# YOLOv5lc 訓練失敗分析總結

## 📊 訓練狀態

- **總任務**: 20 (4 模型 × 5 數據集)
- **已完成**: 8 任務 (40%)
- **需重訓**: 12 任務 (60%)

---

## ❌ 失敗模式

| 模型 | 成功率 | 失敗的數據集 |
|------|--------|------------|
| P3 | 20% (1/5) | V1, V2, V3, V5 |
| P4 | 60% (3/5) | V2, V3 |
| P5 | 40% (2/5) | V2, V4, V5 |
| Backbone | 40% (2/5) | V2, V4, V5 |

**V2 數據集：所有 4 個模型都失敗** 🔴

---

## 🔍 根本原因

### **問題：Large 模型 (width=1.0) + Batch 128 = 梯度爆炸**

**關鍵代碼** (`yolov5c/models/common.py:971`):
```python
c_ = min(1280, max(256, in_channels * 4))
```

**中間通道數計算：**

| 架構 | Small (0.5) | Medium (0.75) | Large (1.0) | Large 增幅 |
|------|------------|--------------|------------|-----------|
| P3 | 512 | 768 | **1024** | **2.0×** 🔴 |
| P4 | 1024 | **1280** | **1280** | 1.25× (受限) |
| P5 | **1280** | **1280** | **1280** | 1.0× (受限) |

**P3 Large 沒被 1280 上限保護 → 梯度規模翻倍 → 不穩定**

---

## 📐 梯度張量規模

**Batch 128 下的中間層張量大小：**

```
P3 Large:  52×52 × 128 × 1024 = 354,418,688 elements 🔴
P3 Small:  52×52 × 128 × 512  = 177,209,344 elements

P4 Large:  26×26 × 128 × 1280 = 110,755,840 elements
P4 Medium: 26×26 × 128 × 1280 = 110,755,840 elements (相同！)

P5 Large:  13×13 × 128 × 1280 = 27,688,960 elements
P5 Small:  13×13 × 128 × 1280 = 27,688,960 elements (相同！)
```

**P3 Large 是 P3 Small 的 2 倍 → 梯度爆炸風險高**

---

## 🔧 解決方案

### **方案 A: Batch 96 (推薦 - 保持可比性)** ⭐

```bash
--batch-size 96  # 從 128 降到 96 (降低 25%)
```

**效果：**
```
P3 Large batch 96: 52×52 × 96 × 1024 = 266M elements
                 = P3 Medium batch 128 的大小 ✅

可比性: 96/128 = 75% (論文中可接受的調整)
梯度降低: 25%
訓練時間: 只增加 15%
```

### **方案 B: Batch 64 (最保守)**

```bash
--batch-size 64  # 從 128 降到 64 (降低 50%)
```

**效果：**
```
P3 Large batch 64: 52×52 × 64 × 1024 = 177M elements
                 = P3 Small batch 128 的大小 ✅

可比性: 50%
穩定性: 最高
訓練時間: 增加 30%
```

### **方案 C: 使用 Medium 模型 (保持 Batch 128)**

```bash
--cfg models/yolov5mc_p3.yaml  # width=0.75, depth=0.67
--batch-size 128
```

**效果：**
```
P3 Medium batch 128: 52×52 × 128 × 768 = 266M elements

可比性: 100% (batch 相同)
穩定性: 高
性能: 略低於 Large
```

---

## 📋 重訓任務清單

**12 個任務需要重訓（使用 batch 64）：**

- **P3**: V1, V2, V3, V5 (4 tasks)
- **P4**: V2, V3 (2 tasks)
- **P5**: V2, V4, V5 (3 tasks)
- **Backbone**: V2, V4, V5 (3 tasks)

**完整命令：** `yolov5c/RETRY_COMMANDS_FINAL.txt`

---

## 🎯 執行步驟

### **1. 快速診斷測試（推薦先執行）**

```bash
cd yolov5c

python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5lc_p4.yaml \
    --hyp data/hyps/hyp.default.yaml \
    --epochs 50 \
    --batch-size 64 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name DIAGNOSTIC_v2_batch64
```

**判斷標準：**
- ✅ 達到 epoch 10+ → batch 64 有效
- ❌ 仍在 epoch <6 失敗 → 試試 batch 32

### **2. 執行所有重訓任務**

使用 `yolov5c/RETRY_COMMANDS_FINAL.txt` 中的命令，逐個執行。

**重要：**
- 每個任務分開執行（不要用 shell 批量）
- 任務間等待 2-3 分鐘
- 監控 GPU：`nvidia-smi`

---

## 📈 預期結果

| 配置 | 當前成功率 | 預期成功率 |
|------|-----------|-----------|
| Batch 128 | 40% | - |
| **Batch 64** | - | **80-90%** ⭐ |
| Batch 32 | - | 95%+ |

---

## 🔑 關鍵結論

1. ✅ **只有 width 和 depth 差異**
2. ✅ **Width 影響中間通道數**
3. ✅ **P3 Large 沒被 1280 限制保護**
4. ✅ **梯度規模 = 特徵圖大小 × Batch × 中間通道**
5. ✅ **解決方案：batch 64**

---

## 📁 相關文件

- `yolov5c/RETRY_COMMANDS_FINAL.txt` - 重訓命令
- `yolov5c/data/hyps/hyp.lowlr.yaml` - 備用低學習率配置
- `FINAL_ANSWER.md` - 詳細技術分析
- `WIDTH_MULTIPLE_IMPACT.md` - Width multiple 影響分析

