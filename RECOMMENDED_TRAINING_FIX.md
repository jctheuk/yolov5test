# 🚀 修復訓練問題 - 推薦配置

## 📊 當前問題總結

**Job 264168_20 結果**:
- ✅ 訓練準確率: **100%**
- ❌ 驗證準確率: **40.88%**
- 🔴 **嚴重過擬合！**

---

## 🔍 問題根源（除了學習率）

### 1. 🔴 優化器選擇不當
- **當前**: SGD + lr0=0.01
- **問題**: SGD 對學習率極其敏感
- **修復**: 使用 Adam + lr0=0.001

### 2. 🔴 訓練太久
- **當前**: 300 epochs
- **問題**: 小數據集上訓練太久導致記憶訓練集
- **修復**: 50-100 epochs + 早停

### 3. 🟡 層凍結可能不當
- **當前**: freeze=[0]
- **問題**: 限制了模型學習能力
- **修復**: freeze=[] (不凍結)

### 4. 🟡 批次大小較小
- **當前**: batch_size=32
- **問題**: 梯度估計不穩定
- **修復**: batch_size=64-128

### 5. 🟡 類別不平衡
- **數據**: A4C(59), PSAX(33), PLAX(89)
- **問題**: PSAX 樣本太少
- **修復**: 可能需要類別權重或重採樣

---

## ✅ 推薦的完整解決方案

### 選項 1: 使用優化的超參數文件（推薦）

```powershell
# 清除快取
Remove-Item regurgitationV1/*/labels/*.cache* -Force -ErrorAction SilentlyContinue

# 使用優化配置訓練
python train_classification_task.py `
    --data regurgitationV1/data.yaml `
    --hyp yolov5c/data/hyps/hyp.classify_optimized.yaml `
    --epochs 100 `
    --batch-size 64 `
    --optimizer Adam `
    --patience 10 `
    --freeze 0 `
    --cos-lr `
    --cache ram `
    --device auto
```

**關鍵改變**:
- ✅ `lr0: 0.001` (降低 10 倍)
- ✅ `optimizer: Adam` (更穩定)
- ✅ `epochs: 100` (減少 3 倍)
- ✅ `patience: 10` (啟用早停)
- ✅ `batch_size: 64` (增加穩定性)
- ✅ `--cos-lr` (餘弦學習率調度)

---

### 選項 2: 完全模仿 YOLOv5 Classify

```powershell
# 使用與成功的 YOLOv5 Classify 完全相同的配置
python train_classification_task.py `
    --data regurgitationV1/data.yaml `
    --epochs 100 `
    --batch-size 128 `
    --optimizer Adam `
    --lr0 0.001 `
    --weight-decay 0.00005 `
    --patience 20 `
    --cache ram `
    --device auto
```

---

### 選項 3: 保守配置（如果 GPU 記憶體有限）

```powershell
python train_classification_task.py `
    --data regurgitationV1/data.yaml `
    --hyp yolov5c/data/hyps/hyp.classify_optimized.yaml `
    --epochs 50 `
    --batch-size 32 `
    --optimizer Adam `
    --patience 10 `
    --device auto
```

---

## 📈 預期改善

### 當前結果（失敗）❌
```
Epoch 298:
├── 訓練損失: 0.00005
├── 訓練準確率: 100%
├── 驗證準確率: 41%
└── 結論: 嚴重過擬合
```

### 預期結果（成功）✅
```
Epoch 40-60:
├── 訓練損失: 0.2-0.3
├── 訓練準確率: 92-95%
├── 驗證準確率: 90-94%
└── 結論: 良好泛化
```

---

## 🎯 監控指標

### ✅ 訓練成功的標誌

1. **損失穩定下降**
   - 訓練損失: 1.0 → 0.3 → 0.2
   - 驗證損失: 1.0 → 0.4 → 0.3

2. **準確率穩定上升**
   - 訓練準確率: 40% → 80% → 95%
   - 驗證準確率: 40% → 75% → 92%

3. **泛化差距小**
   - 訓練 vs 驗證準確率差距 < 5%

4. **早停觸發**
   - 驗證損失不再下降時自動停止
   - 避免過擬合

### ❌ 需要重新調整的標誌

1. **訓練損失接近 0**
   - 表示記憶訓練集

2. **驗證準確率不提升**
   - 表示無法泛化

3. **訓練/驗證差距 > 10%**
   - 表示過擬合

---

## 🔧 進階優化（可選）

### 1. 處理類別不平衡

如果 PSAX 類別表現特別差，可以添加類別權重：

```python
# 在 train_classification_task.py 中添加
class_weights = torch.tensor([1.0, 2.0, 0.8])  # 給 PSAX 更高權重
```

### 2. 調整數據擴增

如果模型仍然過擬合，可以增加數據擴增：

```yaml
# hyp.yaml
degrees: 15.0  # 增加旋轉
scale: 0.2     # 增加縮放
translate: 0.15  # 增加平移
```

### 3. 使用學習率查找

```powershell
# 找出最佳學習率
python train_classification_task.py `
    --data regurgitationV1/data.yaml `
    --lr-find `
    --epochs 5 `
    --device auto
```

---

## 📝 訓練檢查清單

訓練前確認：
- [ ] 清除所有快取文件（*.cache*）
- [ ] 學習率設置為 0.001
- [ ] 優化器設置為 Adam
- [ ] Epochs 設置為 50-100
- [ ] 啟用早停（patience > 0）
- [ ] 批次大小盡可能大（64-128）
- [ ] 檢查類別名稱順序正確

訓練中監控：
- [ ] 訓練損失穩定下降
- [ ] 驗證準確率穩定上升
- [ ] 訓練/驗證差距 < 10%
- [ ] 沒有 NaN 或 Inf 錯誤

訓練後檢查：
- [ ] 驗證準確率 > 90%
- [ ] 混淆矩陣各類別表現均衡
- [ ] 在測試集上表現良好

---

## 🚀 立即行動

**推薦命令（複製貼上即可）**:

```powershell
# 1. 清除快取
Remove-Item regurgitationV1/*/labels/*.cache* -Force -ErrorAction SilentlyContinue

# 2. 開始訓練
python train_classification_task.py `
    --data regurgitationV1/data.yaml `
    --hyp yolov5c/data/hyps/hyp.classify_optimized.yaml `
    --epochs 100 `
    --batch-size 64 `
    --optimizer Adam `
    --patience 10 `
    --cos-lr `
    --cache ram `
    --device auto `
    > training_optimized.log 2>&1

# 3. 實時監控（另開一個終端）
Get-Content training_optimized.log -Wait | Select-String "Epoch|accuracy|mAP"
```

預期在 40-60 epochs 達到 90%+ 驗證準確率！


