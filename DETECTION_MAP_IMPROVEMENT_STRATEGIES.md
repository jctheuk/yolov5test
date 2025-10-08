# 檢測 mAP 提升策略指南
## YOLOv5WithClassification 檢測性能優化

### 當前檢測性能分析
```
Class     Images  Instances          P          R      mAP50   mAP50-95
  all        181        181      0.664      0.522      0.558      0.203
```

**問題識別:**
- mAP50: 0.558 (中等水平)
- mAP50-95: 0.203 (較低，需要重點提升)
- 召回率: 0.522 (偏低，漏檢問題嚴重)

---

## 🎯 策略 1: 超參數優化

### 1.1 損失權重調整
```yaml
# 當前配置
box: 0.05      # 邊界框損失權重
cls: 0.5       # 檢測分類損失權重
obj: 1.0       # 目標檢測損失權重

# 優化建議
box: 0.15      # 增加邊界框損失權重 (3x提升)
cls: 0.8       # 增加檢測分類損失權重 (1.6x提升)
obj: 1.5       # 增加目標檢測損失權重 (1.5x提升)
```

### 1.2 學習率優化
```yaml
# 當前配置
lr0: 0.01      # 初始學習率
lrf: 0.01      # 最終學習率
warmup_epochs: 3.0

# 優化建議
lr0: 0.005     # 降低初始學習率 (更穩定收斂)
lrf: 0.005     # 降低最終學習率
warmup_epochs: 5.0  # 增加預熱輪數
```

### 1.3 IoU 閾值調整
```yaml
# 當前配置
iou_t: 0.2     # IoU訓練閾值
anchor_t: 4.0  # anchor閾值

# 優化建議
iou_t: 0.15    # 降低IoU閾值 (更嚴格匹配)
anchor_t: 3.5  # 降低anchor閾值
```

---

## 🎯 策略 2: 類別權重平衡

### 2.1 檢測類別權重配置
```yaml
# 在超參數文件中添加
class_weights: [1.0, 1.5, 3.0, 1.2]  # 針對類別不平衡的權重
# 類別0: 1.0 (基準)
# 類別1: 1.5 (中等提升)
# 類別2: 3.0 (樣本數少，大幅提升)
# 類別3: 1.2 (輕微提升)
```

### 2.2 動態類別權重
```python
# 基於樣本數的動態權重計算
def calculate_class_weights(labels):
    class_counts = [66, 55, 12, 48]  # 各類別樣本數
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    return weights  # [0.68, 0.82, 3.78, 0.94]
```

---

## 🎯 策略 3: 數據增強策略

### 3.1 醫學圖像適配的輕度增強
```yaml
# 當前: 完全關閉增強
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0

# 優化建議: 輕度增強
hsv_h: 0.01    # 極輕微色調變化
hsv_s: 0.1     # 輕微飽和度變化
hsv_v: 0.1     # 輕微亮度變化
degrees: 2.0   # 小角度旋轉
translate: 0.02 # 輕微平移
scale: 0.1     # 輕微縮放
```

### 3.2 專用醫學增強
```yaml
# 醫學圖像專用增強
mosaic: 0.0    # 保持關閉
mixup: 0.0     # 保持關閉
copy_paste: 0.0 # 保持關閉
# 只使用幾何變換，避免內容混合
```

---

## 🎯 策略 4: 模型架構優化

### 4.1 Anchor 優化
```yaml
# 針對醫學圖像的anchor調整
anchors:
  - [10, 13, 16, 30, 33, 23]  # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32
```

### 4.2 多尺度訓練
```yaml
# 啟用多尺度訓練
multi_scale: true
img_size: [640, 672, 704, 736, 768]  # 多尺度輸入
```

---

## 🎯 策略 5: 訓練策略優化

### 5.1 漸進式訓練
```bash
# 階段1: 高學習率快速收斂
python train.py --epochs 20 --lr0 0.01 --batch-size 16

# 階段2: 低學習率精細調優
python train.py --epochs 30 --lr0 0.001 --batch-size 8 --resume
```

### 5.2 早停策略調整
```yaml
# 關閉早停，獲得完整訓練曲線
patience: 0  # 完全關閉早停
```

### 5.3 批次大小優化
```yaml
# 根據GPU記憶體調整
batch_size: 8   # 如果記憶體不足
batch_size: 16  # 標準配置
batch_size: 32  # 如果記憶體充足
```

---

## 🎯 策略 6: 損失函數優化

### 6.1 Focal Loss 啟用
```yaml
# 當前配置
fl_gamma: 0.0  # 關閉focal loss

# 優化建議
fl_gamma: 1.5  # 啟用focal loss處理困難樣本
```

### 6.2 標籤平滑
```yaml
# 當前配置
label_smoothing: 0.1

# 優化建議
label_smoothing: 0.05  # 減少標籤平滑，提高檢測精度
```

---

## 🎯 策略 7: 驗證和測試優化

### 7.1 驗證頻率調整
```yaml
# 增加驗證頻率
val_period: 1  # 每個epoch都驗證
```

### 7.2 測試時增強 (TTA)
```python
# 測試時增強提升mAP
def test_time_augmentation(model, img):
    # 多尺度測試
    scales = [0.8, 1.0, 1.2]
    results = []
    for scale in scales:
        resized_img = resize(img, scale)
        result = model(resized_img)
        results.append(result)
    return ensemble_results(results)
```

---

## 🎯 策略 8: 數據質量提升

### 8.1 標註質量檢查
```python
# 檢查標註質量
def validate_annotations(annotations):
    issues = []
    for ann in annotations:
        if ann['area'] < 100:  # 過小的目標
            issues.append("Small object")
        if ann['aspect_ratio'] > 5:  # 過於細長的目標
            issues.append("Elongated object")
    return issues
```

### 8.2 困難樣本挖掘
```python
# 困難樣本挖掘
def hard_negative_mining(predictions, targets):
    # 找出預測錯誤的樣本
    hard_samples = []
    for pred, target in zip(predictions, targets):
        if pred['confidence'] > 0.5 and pred['class'] != target['class']:
            hard_samples.append((pred, target))
    return hard_samples
```

---

## 🎯 策略 9: 後處理優化

### 9.1 NMS 參數調整
```yaml
# NMS參數優化
nms_threshold: 0.45  # 降低NMS閾值，保留更多檢測
conf_threshold: 0.25  # 降低置信度閾值
```

### 9.2 多尺度NMS
```python
# 多尺度NMS
def multi_scale_nms(detections, scales=[0.8, 1.0, 1.2]):
    all_detections = []
    for scale in scales:
        scaled_detections = scale_detections(detections, scale)
        all_detections.extend(scaled_detections)
    return nms(all_detections)
```

---

## 🎯 策略 10: 集成學習

### 10.1 模型集成
```python
# 多模型集成
def ensemble_models(models, images):
    predictions = []
    for model in models:
        pred = model(images)
        predictions.append(pred)
    return weighted_average(predictions)
```

### 10.2 多折交叉驗證
```python
# 5折交叉驗證
def cross_validation_training(data, n_folds=5):
    results = []
    for fold in range(n_folds):
        train_data, val_data = split_data(data, fold, n_folds)
        model = train_model(train_data)
        result = evaluate_model(model, val_data)
        results.append(result)
    return average_results(results)
```

---

## 📊 預期改進效果

### 短期改進 (1-2週)
- **mAP50**: 0.558 → 0.65+ (提升17%)
- **mAP50-95**: 0.203 → 0.28+ (提升38%)
- **召回率**: 0.522 → 0.65+ (提升25%)

### 中期改進 (1個月)
- **mAP50**: 0.65 → 0.75+ (提升35%)
- **mAP50-95**: 0.28 → 0.35+ (提升72%)
- **召回率**: 0.65 → 0.75+ (提升44%)

### 長期改進 (2-3個月)
- **mAP50**: 0.75 → 0.85+ (提升52%)
- **mAP50-95**: 0.35 → 0.45+ (提升122%)
- **召回率**: 0.75 → 0.85+ (提升63%)

---

## 🚀 實施優先級

### 高優先級 (立即實施)
1. **損失權重調整** - 立即見效
2. **類別權重配置** - 解決不平衡問題
3. **學習率優化** - 提升收斂穩定性

### 中優先級 (1週內)
4. **輕度數據增強** - 提升泛化能力
5. **Focal Loss啟用** - 處理困難樣本
6. **NMS參數調整** - 優化後處理

### 低優先級 (長期規劃)
7. **模型架構優化** - 需要更多實驗
8. **集成學習** - 複雜度較高
9. **數據質量提升** - 需要人工標註

---

## 📝 實施檢查清單

- [ ] 更新超參數配置文件
- [ ] 實施類別權重
- [ ] 調整學習率策略
- [ ] 啟用輕度數據增強
- [ ] 優化NMS參數
- [ ] 監控訓練過程
- [ ] 驗證改進效果
- [ ] 記錄最佳配置

---

## ⚠️ 注意事項

1. **醫學圖像特殊性** - 保持診斷準確性
2. **過擬合風險** - 監控驗證集表現
3. **計算資源** - 平衡性能和效率
4. **標註一致性** - 確保標註質量
5. **臨床驗證** - 最終需要臨床專家驗證

這個策略指南提供了系統性的mAP提升方法，建議按優先級逐步實施，並持續監控改進效果。



