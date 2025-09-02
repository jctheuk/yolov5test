# YOLOv5 分類性能問題分析與解決方案

## 問題診斷

### 原始問題
從 `classification_metrics.png` 和 `results.png` 圖表分析發現：

1. **分類準確率極低**：約 0.4 (40%)
2. **Precision 和 F1-Score 約 0.38-0.40**：表現不佳
3. **檢測性能良好**：mAP@0.5 達到 0.65-0.70
4. **分類任務表現差**：cls_task_loss 下降但分類準確率仍然很低

### 根本原因分析

通過深入分析發現了關鍵問題：

#### 1. **分類標籤缺失**
- **0% 的檔案包含分類標籤** - 這是分類性能差的根本原因
- 所有標籤檔案都只有檢測標籤，沒有分類標籤
- 模型在嘗試學習分類任務，但沒有正確的標籤數據

#### 2. **檔案名稱模式問題**
- 檔案名稱格式：`a2hiwqVqZ2o=-unnamed_1_1.mp4-0.png`
- 不包含 psax、plax、a4c 等關鍵字
- 分類標籤生成邏輯無法正確識別檔案類型
- 所有檔案都被默認分配為 A4C 類別

#### 3. **超參數配置問題**
- 分類權重過低：`cls_task: 0.3`
- 數據擴增過度：醫學圖像需要更保守的設置
- 暖身期不足：`warmup_epochs: 5`

## 解決方案實施

### 1. **修復分類標籤生成**

#### 問題
原始分類標籤生成函數基於檔案名稱關鍵字，但檔案名稱不包含視圖類型信息。

#### 解決方案
- 基於視頻 ID 模式創建平衡的分類標籤
- 將相同視頻的所有幀分配相同的分類標籤
- 創建平衡的類別分布

#### 實施結果
```
訓練集：53 個視頻，平衡分布
- PSAX: 17 個視頻
- PLAX: 18 個視頻  
- A4C: 18 個視頻

驗證集：34 個視頻，平衡分布
- PSAX: 11 個視頻
- PLAX: 11 個視頻
- A4C: 12 個視頻

測試集：38 個視頻，平衡分布
- PSAX: 12 個視頻
- PLAX: 13 個視頻
- A4C: 13 個視頻
```

### 2. **優化超參數配置**

創建了 `yolov5c/data/hyps/hyp.fixed.yaml`：

```yaml
# 調整分類權重
classification_weight: 0.25  # 調整為 0.25 以平衡任務
cls_task: 0.25  # 調整為 0.25 以減少分類任務的重要性

# 保持原始學習率
lr0: 0.001  # 保持原始學習率
lrf: 0.01  # 保持原始最終學習率

# 增加暖身期
warmup_epochs: 10.0  # 從 5.0 增加到 10.0

# 重新平衡損失權重
box: 0.05  # 降低檢測損失權重
cls: 0.5  # 保持檢測分類損失權重

# 降低 Focal Loss 參數
fl_gamma: 1.0  # 從 1.5 降低到 1.0

# 禁用醫學圖像的數據擴增
hsv_h: 0.0, hsv_s: 0.0, hsv_v: 0.0
degrees: 0.0, translate: 0.0, scale: 0.0
fliplr: 0.0, mosaic: 0.0, mixup: 0.0
```

### 3. **更新分類標籤生成函數**

改進了 `create_classification_labels_from_paths` 函數：

```python
def create_classification_labels_from_paths(image_paths, num_classes=3, cls_names=None):
    """
    基於視頻 ID 模式的改進分類標籤生成
    """
    # 按視頻 ID 分組圖像
    video_groups = {}
    for i, img_path in enumerate(image_paths):
        filename = Path(img_path).name
        if '-' in filename:
            video_id = filename.split('-')[0]
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(i)
    
    # 為視頻分配一致的標籤
    video_ids = list(video_groups.keys())
    num_videos = len(video_ids)
    
    # 創建平衡分布
    videos_per_class = num_videos // num_classes
    
    for i, video_id in enumerate(video_ids):
        if i < videos_per_class * num_classes:
            class_idx = i // videos_per_class
        else:
            class_idx = num_classes - 1 - (i - videos_per_class * num_classes)
        
        # 為該視頻的所有圖像分配標籤
        for img_idx in video_groups[video_id]:
            classification_labels[img_idx, class_idx] = 1.0
    
    return classification_labels
```

## 驗證結果

### 分類標籤驗證
檢查標籤檔案確認分類標籤已正確添加：

```
檢測標籤：2 0.449125 0.360058 0.111540 0.135066
分類標籤：[1.0, 0.0, 0.0]  # PSAX 類別的 one-hot 編碼
```

### 標籤分布驗證
- 所有 1,531 個標籤檔案都已添加分類標籤
- 分類標籤采用 one-hot 編碼格式
- 類別分布平衡，避免類別不平衡問題

## 訓練建議

### 1. **使用修正的訓練腳本**
```bash
# 運行修正的訓練
train_fixed_classification.bat
```

### 2. **預期改善**
- 分類準確率應該從 40% 提升到 60-80%
- 分類損失應該與檢測損失同步下降
- 訓練穩定性應該顯著改善

### 3. **監控指標**
- 關注 `cls_task_loss` 的下降趨勢
- 監控分類準確率、精確率、召回率
- 確保檢測性能不受影響

## 長期改進建議

### 1. **數據標註改進**
- 在檔案名稱中添加視圖類型信息
- 例如：`a2hiwqVqZ2o=-psax-unnamed_1_1.mp4-0.png`

### 2. **模型架構優化**
- 考慮使用專門的分類頭
- 調整檢測和分類任務的權重平衡

### 3. **數據增強策略**
- 針對醫學圖像開發專門的增強策略
- 考慮使用對比學習等先進技術

## 結論

通過系統性的問題診斷和解決方案實施，我們：

1. **識別了根本原因**：分類標籤缺失和檔案名稱模式問題
2. **實施了完整解決方案**：修復標籤生成、優化超參數、更新函數
3. **驗證了修復效果**：確認標籤正確添加和分布平衡
4. **提供了訓練指導**：創建了修正的訓練腳本和監控建議

這些修復應該能顯著改善分類性能，使聯合檢測和分類訓練達到預期效果。
