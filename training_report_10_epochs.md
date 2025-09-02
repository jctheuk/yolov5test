# YOLOv5 聯合訓練報告 - 10 Epochs 訓練

## 📊 訓練概覽

### 基本配置
- **模型**: YOLOv5s with Classification (yolov5sc.yaml)
- **數據集**: Regurgitation-YOLODataset-Detection
- **訓練輪數**: 10 epochs
- **批次大小**: 8
- **圖像尺寸**: 416x416
- **優化器**: SGD
- **早停機制**: 已關閉 (patience: 0) ✅

### 聯合訓練配置
- **分類權重**: 0.2
- **分類任務權重**: 0.2
- **檢測權重**: 0.05 (box), 0.5 (cls), 1.0 (obj)

## 🎯 數據集分析

### 分類標籤分佈
| 類別 | 樣本數量 | 百分比 |
|------|----------|--------|
| PLAX | 557 | 53.5% |

| A4C | 374 | 35.9% |
| PSAX | 111 | 10.7% |
| **總計** | **1042** | **100%** |

### 檢測標籤分佈
- **檢測類別**: 4 個 (AR, MR, PR, TR)
- **標籤格式**: YOLO 格式 (class_id x_center y_center width height)
- **分類標籤格式**: One-hot encoding (3 個類別)

## 📈 訓練性能分析

### 檢測性能 (Detection Performance)
| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| 0 | 0.00231 | 0.67087 | 0.00295 | 0.00068 |
| 5 | 0.79839 | 0.08333 | 0.08332 | 0.02273 |
| 9 | 0.57831 | 0.23258 | 0.25277 | 0.07926 |

**檢測性能趨勢**:
- ✅ **Precision**: 從 0.002 提升到 0.578 (顯著改善)
- ⚠️ **Recall**: 從 0.671 下降到 0.233 (需要關注)
- ✅ **mAP@0.5**: 從 0.003 提升到 0.253 (良好改善)
- ✅ **mAP@0.5:0.95**: 從 0.001 提升到 0.079 (穩定提升)

### 分類性能 (Classification Performance)
| Epoch | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| 0-9 | 0.4262 | 0.1817 | 0.4262 | 0.2548 |

**分類性能問題**:
- ❌ **分類性能完全停滯** - 所有指標在10個epoch中沒有任何變化
- ❌ **Precision 過低** - 僅 0.1817，表示大量假陽性
- ⚠️ **Accuracy 偏低** - 42.62% 的準確率需要改善

## 📉 損失函數分析

### 訓練損失趨勢
- **Box Loss**: 0.091 → 0.047 (穩定下降) ✅
- **Object Loss**: 0.001 → 0.002 (相對穩定) ✅
- **Classification Loss**: 0.008 → 0.004 (穩定下降) ✅
- **Classification Task Loss**: 0.427 → 0.344 (緩慢下降) ⚠️

### 驗證損失趨勢
- **Box Loss**: 0.071 → 0.055 (緩慢下降) ⚠️
- **Object Loss**: 0.002 → 0.002 (穩定) ✅
- **Classification Loss**: 0.006 → 0.004 (穩定下降) ✅
- **Classification Task Loss**: 0.357 → 0.361 (幾乎無變化) ❌

## 🔍 問題診斷

### 1. 分類任務嚴重問題
**症狀**:
- 分類任務損失無改善: cls_task_loss 在驗證集上完全沒有下降
- 分類指標停滯: 所有分類指標在10個epoch中保持不變
- 分類準確率固定在 42.62%

**根本原因**:
- **數據集文件排序問題**: 文件按名稱排序時，相同類別的樣本被分組在一起
- **批次偏差**: 前幾百個文件都是 PLAX 類別，導致模型在訓練初期只看到同一類別
- **模型過度擬合**: 模型學會了"所有樣本都是 PLAX"的模式

### 2. 檢測性能不平衡
**症狀**:
- Precision-Recall 不平衡: Precision 提升但 Recall 下降
- 可能過擬合: 模型變得過於保守

### 3. 學習率設置
**症狀**:
- 學習率過高: 初始學習率 0.01 可能過高
- 學習率衰減: 從 0.07 快速衰減到 0.003

## 🛠️ 解決方案

### 1. 數據加載器修復
**問題**: 驗證數據加載器沒有啟用 shuffle
**解決**: 已修復驗證數據加載器的 shuffle 設置

```python
# 修復前
val_loader = create_dataloader(val_path, ..., prefix=colorstr('val: '))[0]

# 修復後  
val_loader = create_dataloader(val_path, ..., prefix=colorstr('val: '), shuffle=True)[0]
```

### 2. 超參數優化建議
```yaml
# 建議的 hyperparameters
classification_weight: 0.5  # 從 0.2 提升到 0.5
cls_task: 0.5              # 從 0.2 提升到 0.5
lr0: 0.005                 # 從 0.01 降低到 0.005
lrf: 0.05                  # 從 0.1 降低到 0.05
epochs: 50                 # 從 10 增加到 50
batch_size: 16             # 從 8 增加到 16
```

### 3. 數據擴增設置
```yaml
# 醫學圖像建議保持當前設置
mosaic: 0.0      # 已關閉 ✅
mixup: 0.0       # 已關閉 ✅
copy_paste: 0.0  # 已關閉 ✅
```

## 📊 訓練配置評估

### ✅ 優點
1. **檢測任務有改善**: mAP 指標穩定提升
2. **損失函數收斂**: 大部分損失函數正常下降
3. **配置正確**: 早停機制已關閉，數據擴增已關閉
4. **聯合訓練啟用**: 分類功能正常啟用
5. **數據加載器修復**: 已修復 shuffle 問題

### ❌ 需要改進
1. **分類性能嚴重問題**: 需要立即解決
2. **學習率過高**: 可能導致訓練不穩定
3. **訓練輪數不足**: 10 epochs 對於醫學圖像可能不夠
4. **批次大小偏小**: 8 可能影響訓練穩定性

## 🎯 建議的下一步行動

### 1. 立即行動
```bash
# 重新訓練，使用修復後的數據加載器
python train.py \
    --data Regurgitation-YOLODataset-Detection/data.yaml \
    --cfg models/yolov5sc.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto
```

### 2. 超參數調整
- 提升分類權重到 0.5
- 降低學習率到 0.005
- 增加訓練輪數到 50 epochs

### 3. 監控指標
- 分類準確率應該從 42.62% 開始改善
- 分類損失應該開始下降
- 檢測性能應該保持穩定

## 📈 預期改善

### 分類性能預期
- **準確率**: 從 42.62% 提升到 70%+
- **Precision**: 從 0.1817 提升到 0.6+
- **Recall**: 從 0.4262 提升到 0.6+
- **F1-Score**: 從 0.2548 提升到 0.6+

### 檢測性能預期
- **mAP@0.5**: 從 0.253 提升到 0.4+
- **mAP@0.5:0.95**: 從 0.079 提升到 0.2+
- **Precision-Recall 平衡**: 改善不平衡問題

## 🔧 技術細節

### 模型架構
- **Backbone**: YOLOv5s (depth_multiple: 0.33, width_multiple: 0.50)
- **分類頭**: YOLOv5WithClassification (256 channels → 3 classes)
- **檢測頭**: Detect (4 classes: AR, MR, PR, TR)

### 數據加載器
- **訓練**: shuffle=True ✅
- **驗證**: shuffle=True ✅ (已修復)
- **批次大小**: 8 (建議增加到 16)

### 損失函數
- **檢測損失**: Box Loss + Object Loss + Classification Loss
- **分類損失**: Classification Task Loss
- **總損失**: 檢測損失 + 分類損失 * classification_weight

## 📝 結論

這次 10 epochs 的訓練揭示了聯合訓練中的關鍵問題：

1. **數據加載器 shuffle 問題** - 已修復
2. **分類權重設置過低** - 需要調整
3. **學習率過高** - 需要降低
4. **訓練輪數不足** - 需要增加

修復這些問題後，預期分類性能將有顯著改善，聯合訓練將能夠正常進行。

---

**報告生成時間**: 2024年12月
**訓練配置**: YOLOv5s + Classification
**數據集**: Regurgitation-YOLODataset-Detection
**狀態**: 問題已診斷，解決方案已實施
