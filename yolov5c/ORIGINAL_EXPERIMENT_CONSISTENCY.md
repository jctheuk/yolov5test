# 與原始實驗的一致性說明

## 🎯 為什麼移除 `--cos-lr` 和 `--amp`？

### 原始實驗參數檢查

我檢查了你的原始實驗腳本發現：

**你之前使用的參數**：
```bash
python train.py \
  --data ../regurgitationV1/data.yaml \
  --cfg models/yolov5lc_classify_backbone.yaml \
  --epochs 300 \
  --batch-size 64 \
  --imgsz 416 \
  --name yolov5lc_backbone_v1 \
  --cache \
  --nosave \
  --patience 0 \
  --hyp data/hyps/hyp.default.yaml
  # 沒有 --cos-lr
  # 沒有 --amp
```

### `--cos-lr` 功能說明

`--cos-lr` (Cosine Learning Rate) 是餘弦退火學習率調度：

```python
# 學習率變化
固定學習率:    lr = lr0 (固定值)
餘弦退火:      lr = lr0 * (1 + cos(π * epoch / max_epoch)) / 2

# 效果圖
lr0 ----\
         \
          \___
              \___
                  \____> lrf (final lr)
```

**好處**：學習率平滑下降，可能幫助收斂  
**缺點**：改變了實驗條件，無法與原始結果對比

## ✅ **保持實驗一致性的決定**

### 為什麼移除這些參數？

1. **實驗對比需要**：只改變模型大小，其他條件相同
2. **原始未使用**：你之前的成功實驗 (V1, V3) 沒有用這些參數
3. **純淨測試**：專注測試模型大小對記憶體的影響
4. **結果可比性**：可以直接比較 MLC vs 原始 YOLOv5l

### 更新後的 MLC 配置

**完全匹配原始實驗**：
```bash
python train.py \
  --data ../regurgitationV1/data.yaml \
  --cfg models/yolov5mlc_classify_backbone.yaml \  # 僅此行改變
  --epochs 300 \
  --batch-size 64 \
  --imgsz 416 \
  --name yolov5mlc_backbone_v1 \
  --cache \
  --nosave \
  --patience 0 \
  --hyp data/hyps/hyp.default.yaml
  # 與原始實驗完全相同的參數
```

## 📊 實驗對比矩陣

| 項目 | 原始 YOLOv5l | YOLOv5MLC | 對比性 |
|------|-------------|-----------|--------|
| **超參數** | hyp.default.yaml | hyp.default.yaml | ✅ 相同 |
| **學習率** | 固定 (無 --cos-lr) | 固定 (無 --cos-lr) | ✅ 相同 |
| **精度** | FP32 (無 --amp) | FP32 (無 --amp) | ✅ 相同 |
| **批次大小** | 64/128 | 64/128 | ✅ 相同 |
| **訓練設定** | epochs 300, patience 0 | epochs 300, patience 0 | ✅ 相同 |
| **唯一差異** | width=1.0, depth=1.0 | width=0.875, depth=0.83 | ⚠️ 模型大小 |

### 實驗價值

這樣的設置讓你能夠**純淨地測試**：
- 模型大小對記憶體使用的影響
- 模型大小對訓練穩定性的影響  
- 模型大小對最終性能的影響

## 🔬 科學實驗設計

### 控制變量
- ✅ **超參數相同**: 學習率、優化器、增強等
- ✅ **訓練設置相同**: epochs、early stop、快取等
- ✅ **數據相同**: regurgitationV1-V5 相同數據集
- ⚠️ **僅模型不同**: 大小從 Large → Medium-Large

### 預期發現
1. **記憶體使用**: 31.9GB → 22-26GB
2. **訓練穩定性**: 從 40% 成功率 → 95%+ 成功率
3. **模型性能**: 可能略低但更穩定
4. **訓練速度**: 可能稍快 (較小模型)

## 💡 如果之後想實驗進階優化

在完成 MLC 基準實驗後，可以創建進階版本：
- **YOLOv5MLC + --cos-lr**: 測試學習率調度影響
- **YOLOv5MLC + --amp**: 測試混合精度影響
- **YOLOv5MLC + 兩者**: 測試組合優化效果

但現在先保持實驗純淨性，專注解決記憶體問題！

## 📋 結論

移除 `--cos-lr` 和 `--amp` 是正確的決定：
- ✅ 保持與原始實驗的一致性
- ✅ 專注測試模型大小的影響
- ✅ 結果更具可比性
- ✅ 實驗設計更科學

你的 YOLOv5MLC 實驗現在可以直接與原始失敗的 YOLOv5l 實驗進行對比分析了！

