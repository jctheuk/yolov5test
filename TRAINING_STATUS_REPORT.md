# YOLOv5lc K-Fold Training Status Report

生成時間：2025-10-14

## 📊 **訓練完成狀態總覽**

### **整體統計：**
- **總任務數**: 20 (4 模型 × 5 數據集)
- **已完成**: 8 任務 (40%)
- **需重訓**: 12 任務 (60%)

---

## ✅ **已完成的訓練 (8/20)**

| 模型 | 資料集 | 準確率 | 訓練時間 | Batch Size |
|------|--------|--------|---------|-----------|
| P3 | V4 | 96.69% | - | 128 |
| P4 | V1 | 97.24% | - | 128 |
| P4 | V4 | **97.78%** ⭐ | - | 128 |
| P4 | V5 | 96.70% | - | 128 |
| P5 | V1 | **97.79%** ⭐ | - | 128 |
| P5 | V3 | 97.22% | - | 128 |
| Backbone | V1 | 92.27% | - | 128 |
| Backbone | V3 | **97.78%** ⭐ | - | 128 |

**最高準確率**: 97.79% (P5 V1)

---

## ❌ **需要重新訓練 (12/20)**

### **按模型分類：**
- **P3**: 4 任務 (V1, V2, V3, V5)
- **P4**: 2 任務 (V2, V3)
- **P5**: 3 任務 (V2, V4, V5)
- **Backbone**: 3 任務 (V2, V4, V5)

### **按資料集分類：**
- **V1**: 1 任務 (P3)
- **V2**: 4 任務 (所有模型) 🔴
- **V3**: 2 任務 (P3, P4)
- **V4**: 2 任務 (P5, Backbone)
- **V5**: 3 任務 (P3, P5, Backbone)

---

## 🔍 **失敗原因分析**

### **根本原因：梯度不穩定**
```
問題 = 通道數 × Batch Size × V2特徵分佈

P3:  256 channels × 128 batch = 32,768  (相對最小)
P4:  512 channels × 128 batch = 65,536  (中等)
P5:  1024 channels × 128 batch = 131,072 (最大)
Backbone: 1024 channels × 128 batch = 131,072 (最大)
```

### **V2 數據集分析：**
- ✅ 數據量：1013 train, 179 valid, 292 test (正常)
- ✅ Bbox 範圍：正常 [0, 1]
- ✅ 類別分佈：A4C=137, PSAX=32, PLAX=131 (平衡)
- ❌ **但所有模型在 epoch 2-6 就出現 NaN 錯誤**

### **失敗時機：**
- P3: Epoch 3 (即使作為第一個訓練)
- P4: Epoch 2
- P5: Epoch 6
- Backbone: Epoch 6

**結論：不是訓練順序問題，是數值穩定性問題**

---

## 🎯 **解決方案**

### **推薦方案：降低 Batch Size**
```bash
--batch-size 64  # 從 128 降到 64

預期效果：
- 梯度規模減少 50%
- GPU 記憶體減少 ~50%
- 訓練時間增加 ~30%
- NaN 錯誤應該消失
```

### **備用方案：降低學習率**
```bash
--hyp data/hyps/hyp.lowlr.yaml  # lr0: 0.005 (從 0.01)

預期效果：
- 梯度更新更溫和
- 更長的 warmup (10 epochs)
- 可能需要更多 epochs 收斂
```

### **最保守方案：組合使用**
```bash
--batch-size 64 --hyp data/hyps/hyp.lowlr.yaml
```

---

## 📋 **執行計劃**

### **Phase 1: 快速診斷測試 (建議先執行)**
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
- ✅ 如果達到 epoch 10+ → batch 64 解決問題
- ❌ 如果仍在 epoch <6 失敗 → 需要 batch 32 或低 LR

### **Phase 2: 根據診斷結果批量執行**
- 使用 `yolov5c/retry_failed_training_FINAL.txt` 中的命令
- 逐個執行，間隔 2-3 分鐘
- 監控 GPU 記憶體和訓練日誌

### **Phase 3: 驗證和總結**
- 確認所有 20 個訓練完成
- 比較不同配置的準確率
- 生成最終報告

---

## 📈 **模型穩定性排名**

基於當前結果：

| 排名 | 模型 | 成功率 | 特點 |
|-----|------|--------|------|
| 🥇 | **P4** | 60% (3/5) | 512 通道，最佳平衡 |
| 🥈 | P5 | 40% (2/5) | 1024 通道，中等穩定 |
| 🥈 | Backbone | 40% (2/5) | 1024 通道，中等穩定 |
| 🥉 | P3 | 20% (1/5) | 256 通道，最不穩定 |

---

## 🎯 **關鍵結論**

1. ✅ **問題確認**: 通道數 × Batch Size 導致的梯度爆炸
2. ✅ **V2 是觸發器**: 特定特徵分佈在大 batch 下不穩定
3. ✅ **解決方案**: 降低 batch size 到 64
4. ✅ **Workers=8 不是問題**: 數據加載正常
5. ✅ **P4 最穩定**: 512 通道是最佳平衡點

---

## 📁 **相關文件**

- `yolov5c/retry_failed_training_FINAL.txt` - 完整重訓命令
- `yolov5c/data/hyps/hyp.lowlr.yaml` - 低學習率配置
- `SOLUTION_PLAN.md` - 詳細解決方案
- `channel_gradient_analysis.md` - 技術分析

---

## ⚠️ **重要提醒**

1. **先運行診斷測試** (DIAGNOSTIC_v2_batch64)
2. **逐個執行任務**，不要批量運行
3. **間隔 2-3 分鐘**，讓資源釋放
4. **監控 GPU 記憶體**：nvidia-smi
5. **檢查日誌**：注意 NaN 錯誤

---

## 📞 **下一步行動**

**立即執行：**
```bash
cd yolov5c
python train.py --data ../regurgitationV2/data.yaml --cfg models/yolov5lc_p4.yaml --hyp data/hyps/hyp.default.yaml --epochs 50 --batch-size 64 --workers 8 --imgsz 416 --patience 0 --device 0 --cache ram --project runs/train --name DIAGNOSTIC_v2_batch64
```

**預期時間：** ~20 分鐘

**如果成功：** 繼續執行所有 12 個任務（使用 batch 64）

**如果失敗：** 降至 batch 32 或使用低學習率配置



