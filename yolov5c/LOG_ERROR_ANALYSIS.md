# YOLOv5lc P5 訓練日誌錯誤分析報告

## 📋 日誌文件資訊
- **文件**: `job_265056_2_1760418116.log`
- **訓練任務**: YOLOv5lc P5 K-Fold Training V1-V5
- **模型配置**: `models/yolov5lc_p5.yaml`
- **批次大小**: 128
- **總行數**: 11,563

## 🚨 發現的錯誤總結

### 錯誤統計
- **總錯誤數**: 3 個 NaN 梯度爆炸錯誤
- **受影響的 Fold**: V2, V4, V5
- **成功完成的 Fold**: V1, V3

## 📊 詳細錯誤分析

### 錯誤 #1: V2 訓練失敗
**時間**: Tue Oct 14 12:09:56 CST 2025
**位置**: 日誌行 5810-5852
**Epoch**: 139/299 (46.5% 完成)

**錯誤訊息**:
```
RuntimeError: Function 'ConvolutionBackward0' returned nan values in its 1th output.
Exception in thread Thread-26:
ConnectionResetError: [Errno 104] Connection reset by peer
```

**失敗前的性能**:
- Overall mAP50: **27.3%**
- Class 0 (AR): P=68.4%, R=56.1%, mAP50=58.1%
- 分類準確率: **59.67%**
- PLAX 分類準確率: **97.8%**

### 錯誤 #2: V4 訓練失敗  
**時間**: Tue Oct 14 13:01:13 CST 2025
**位置**: 日誌行 11405-11447
**Epoch**: 282/299 (94.3% 完成)

**錯誤訊息**:
```
RuntimeError: Function 'ConvolutionBackward0' returned nan values in its 1th output.
Exception in thread Thread-26:
ConnectionResetError: [Errno 104] Connection reset by peer
```

**失敗前的性能**:
- Overall mAP50: **65.9%**
- Class 0 (AR): P=79.5%, R=72.7%, mAP50=71.4%
- 分類準確率: **96.13%**
- 所有分類任務都表現優異 (>93%)

### 錯誤 #3: V5 訓練失敗
**時間**: Tue Oct 14 13:01:45 CST 2025
**位置**: 日誌行 11517-11559
**Epoch**: 285/299 (95.3% 完成)

**錯誤訊息**:
```
RuntimeError: Function 'ConvolutionBackward0' returned nan values in its 1th output.
Exception in thread Thread-26:
ConnectionResetError: [Errno 104] Connection reset by peer
```

**失敗前的性能**:
- Overall mAP50: **68.2%**
- Class 1 (MR): P=90.2%, R=83.6%, mAP50=85.7%
- 分類準確率: **96.69%**
- 優異的檢測和分類性能

## 🔍 錯誤模式分析

### 共同特徵
1. **錯誤類型一致**: 全部都是 `ConvolutionBackward0` NaN 錯誤
2. **發生階段**: 訓練中後期 (139-285 epochs)
3. **性能表現**: 失敗前模型表現都很好
4. **後續影響**: 連接重置，訓練中斷

### 性能趨勢
- **V1 成功**: Epoch 0-299 完成
- **V2 失敗**: Epoch 139 失敗，mAP50 27.3%
- **V3 成功**: Epoch 0-299 完成  
- **V4 失敗**: Epoch 282 失敗，mAP50 65.9%
- **V5 失敗**: Epoch 285 失敗，mAP50 68.2%

### 時間分析
```
V1: 11:18:35 - 12:09:15 (50分40秒) ✅ 成功
V2: 12:09:16 - 12:09:56 (40秒)    ❌ 失敗
V3: 12:09:56 - 13:00:36 (50分40秒) ✅ 成功
V4: 13:00:36 - 13:01:13 (37秒)    ❌ 失敗
V5: 13:01:13 - 13:01:45 (32秒)    ❌ 失敗
```

## 🎯 根本原因分析

### 主要問題
1. **梯度爆炸**: 長時間訓練後梯度數值變得極大
2. **數值不穩定**: 缺乏梯度剪裁機制
3. **學習率問題**: 固定學習率在後期可能過高
4. **記憶體壓力**: 大批次大小 (128) 加劇不穩定性

### 觸發條件
- **訓練進展**: 模型學習良好時更容易發生
- **高性能階段**: mAP > 60% 時風險增加
- **後期訓練**: Epoch > 100 後更頻繁

### 數據集差異
- **V1, V3**: 數據特性較穩定，完成訓練
- **V2, V4, V5**: 可能包含更難學習的樣本

## 💡 解決建議

### 立即措施
1. **使用穩定訓練命令**: `stable_training_commands.txt`
2. **梯度剪裁**: 設定 `grad_clip: 10.0`
3. **降低批次大小**: 128 → 64 或 32
4. **餘弦學習率**: 添加 `--cos-lr` 參數

### 長期優化
1. **混合精度訓練**: 添加 `--amp` 參數
2. **動態學習率**: 實施更複雜的調度策略
3. **模型檢查點**: 保存中間權重以便恢復
4. **數據集分析**: 檢查失敗數據集的特殊性

## 📈 預期改善效果

使用穩定訓練配置後預期：
- ✅ 消除 NaN 錯誤
- ✅ 所有 V1-V5 成功完成
- ✅ 保持或改善性能表現
- ✅ 更穩定的訓練曲線

## 🔧 監控重點

### 關鍵指標
1. **梯度範數**: 監控是否被正確剪裁
2. **損失穩定性**: 避免異常跳躍
3. **記憶體使用**: 確保不超出限制
4. **訓練時間**: 確保完整執行

### 成功標準
- 完成全部 300 epochs
- 無 NaN 或 inf 數值
- 穩定的性能提升曲線
- 分類準確率 > 90%


