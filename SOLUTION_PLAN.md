# YOLOv5lc Training Failure Solution Plan

## 📊 **問題確認**

### **根本原因：通道數 × Batch Size 導致的梯度不穩定**

您的診斷完全正確！問題是：
```
梯度規模 = 通道數 × Batch Size × V2特徵值分佈
```

### **證據：**
1. ✅ **所有架構在 V2 上都在早期失敗** (epoch 2-6)
2. ✅ **P3 即使作為第一個訓練也失敗** (排除訓練順序問題)
3. ✅ **通道數不同但都失敗** (256, 512, 1024)
4. ✅ **V2 數據統計正常** (bbox, 類別分佈都正常)

### **結論：**
V2 的某些樣本特徵值分佈，在 **batch-size 128** 下會導致梯度爆炸。

---

## 🔧 **解決方案（按優先順序）**

### **方案 1: 降低 Batch Size** ⭐ 推薦優先測試
```bash
# 測試：P4 + V2 + batch 64
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
    --name test_v2_batch64

# 如果成功 → 用 batch 64 重新訓練所有 V2 失敗任務
# 如果失敗 → 試試 batch 32
```

### **方案 2: 降低學習率**
```bash
# 測試：P4 + V2 + 低學習率
python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5lc_p4.yaml \
    --hyp data/hyps/hyp.lowlr.yaml \
    --epochs 50 \
    --batch-size 128 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name test_v2_lowlr

# hyp.lowlr.yaml 已創建 (lr0=0.005, warmup=10)
```

### **方案 3: 組合方案（最保守）**
```bash
# Batch 64 + 低學習率
python train.py \
    --data ../regurgitationV2/data.yaml \
    --cfg models/yolov5lc_p4.yaml \
    --hyp data/hyps/hyp.lowlr.yaml \
    --epochs 50 \
    --batch-size 64 \
    --workers 8 \
    --imgsz 416 \
    --patience 0 \
    --device 0 \
    --cache ram \
    --project runs/train \
    --name test_v2_batch64_lowlr
```

---

## 📋 **完整任務清單（基於測試結果）**

### **待測試任務：**
1. ✅ 創建 `hyp.lowlr.yaml` - 完成
2. 🔄 運行 Test 1 (batch 64)
3. 🔄 根據結果決定策略

### **測試後的行動：**

#### **如果 batch 64 解決問題：**
```bash
# 用 batch 64 重新訓練所有 V2 失敗的任務
--batch-size 64  # 應用到所有 V2 相關訓練
```

#### **如果需要更低的 batch：**
```bash
# 用 batch 32
--batch-size 32
```

#### **如果需要降低學習率：**
```bash
--hyp data/hyps/hyp.lowlr.yaml  # 應用到所有問題訓練
```

---

## 🎯 **為什麼其他數據集（V1, V3, V4, V5）可以用 batch 128？**

### **假設：**
1. **V2 的特徵分佈更極端**
   - 可能包含更多邊緣案例
   - 特徵值的方差更大
   - 在大 batch 下梯度累積更不穩定

2. **V2 的 K-Fold 分割剛好集中了困難樣本**
   - V2 train 有 1013 個樣本（最多）
   - 可能包含更多難樣本的組合

3. **類別分佈的微妙差異**
   - V2: A4C=137, PSAX=32, PLAX=131
   - V4: A4C=124, PSAX=30, PLAX=146
   - PSAX 最少可能導致類別不平衡效應放大

---

## 📊 **測試矩陣**

| 測試 | Batch | LR | 預期 GPU 記憶體 | 執行時間 | 如果成功則說明 |
|------|-------|----|--------------|---------| --------------|
| Test 1 | 64 | 0.01 | ~16-20G | 50 epochs ~20min | Batch size 問題 |
| Test 2 | 128 | 0.005 | ~32G | 50 epochs ~15min | 學習率問題 |
| Test 3 | 64 | 0.005 | ~16-20G | 50 epochs ~20min | 組合問題 |
| Test 4 | 32 | 0.01 | ~8-12G | 50 epochs ~30min | 嚴重的 batch 問題 |

---

## 🎯 **立即行動項目**

### **Step 1: 快速測試 (50 epochs)**
```bash
cd yolov5c

# 最可能成功的配置
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
    --name quick_test_v2_batch64

# 監控命令：
# nvidia-smi -l 1  # 監控 GPU
# tail -f runs/train/quick_test_v2_batch64/results.csv  # 監控訓練
```

### **Step 2: 根據結果更新所有命令**
- 如果成功 → 使用 batch 64 重新生成所有重試命令
- 如果失敗 → 測試 batch 32 或低學習率

---

## 💡 **最終答案：**

**問題確實是：梯度問題，由通道數 × Batch Size 引起**

**最可能的解決方案：降低 batch size 到 64 或 32**

建議立即運行 Step 1 的快速測試來驗證！






