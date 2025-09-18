# YOLOv5 問題與解決方案總結

## 📋 問題概述

基於 `.specstory` 歷史記錄分析，以下是您在 YOLOv5 聯合訓練項目中遇到的主要問題和解決方案：

---

## 🔍 主要問題識別

### 1. **過擬合問題 (Overfitting)**
- **問題描述**: 檢測 mAP 表現良好，但分類任務出現過擬合
- **具體表現**: 
  - `val/obj_loss` 在 50 epochs 後開始上升 (0.011 → 0.013)
  - 分類準確率出現異常的 100% 虛假準確率
  - 訓練曲線劇烈波動，存在梯度問題

### 2. **類別不平衡問題 (Class Imbalance)**
- **檢測任務分布**:
  - Class 0 (AR): 49.7% - 多數類
  - Class 1 (MR): 31.0% - 中等類  
  - Class 2 (PR): 4.0% - 少數類 ⚠️
  - Class 3 (TR): 15.3% - 少數類 ⚠️

- **分類任務分布** (regurgitationV1):
  - Class 0 (A4C): 32.2% - 中等類
  - Class 1 (PSAX): 20.9% - 少數類 ⚠️
  - Class 2 (PLAX): 46.9% - 多數類

### 3. **訓練不穩定問題**
- **NaN 錯誤**: `CudnnBatchNormBackward0` 返回 NaN 值
- **梯度爆炸**: 學習率過高導致數值不穩定
- **批次正規化問題**: 批次統計計算出現問題

### 4. **技術實現問題**
- **編碼錯誤**: YAML 文件讀取時的字符編碼錯誤
- **PowerShell 顯示問題**: 終端出現字符編碼和顯示問題
- **Optuna 試驗失敗**: 所有 15 個試驗都返回 0.0 分數

---

## 🛠️ 解決方案實施

### 1. **Focal Loss 實現 (分類任務)**

#### **問題**: 分類任務中的類別不平衡導致模型只猜測多數類
#### **解決方案**: 實現針對分類任務的 Focal Loss

```yaml
# hyp.classV1.yaml 配置
cls_focal_gamma: 2.0  # focal loss gamma for classification task
cls_focal_alpha: [0.3, 0.5, 0.2]  # [A4C, PSAX, PLAX] - 針對實際分布
```

#### **權重分配邏輯**:
- **PSAX (0.5)**: 最高權重，因為是少數類 (20.9%)
- **A4C (0.3)**: 中等權重，中等類別 (32.2%)
- **PLAX (0.2)**: 最低權重，多數類別 (46.9%)

#### **實現位置**: `yolov5c/utils/loss.py`
```python
def focal_loss_classification(self, probs, targets):
    # 計算 Focal Loss 來處理分類任務的類別不平衡
    # 自動關注難分類樣本，減少對容易分類樣本的關注
```

### 2. **醫學圖像最佳實踐配置**

#### **問題**: 醫學圖像需要保持原始特徵，避免數據擴增干擾
#### **解決方案**: 創建無數據擴增的超參數配置

```yaml
# hyp.classV1.yaml - 完全關閉數據擴增
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
```

### 3. **NaN 錯誤修復**

#### **問題**: `CudnnBatchNormBackward0` 返回 NaN 值
#### **解決方案**: 增強 NaN 檢測和處理機制

```python
# yolov5c/train.py 中的改進
# 在 backward() 前檢查 loss 是否為 NaN/Inf
if torch.isnan(total_loss) or torch.isinf(total_loss):
    LOGGER.warning(f'NaN/Inf loss detected: {total_loss.item()}, skipping backward pass')
    optimizer.zero_grad()
    continue

# 捕獲 NaN 相關的 RuntimeError
try:
    scaler.scale(total_loss).backward()
except RuntimeError as e:
    if 'nan' in str(e).lower():
        LOGGER.warning(f'NaN detected during backward pass: {e}')
        optimizer.zero_grad()
        scaler.update()  # Reset scaler state
        continue
```

### 4. **超參數優化系統**

#### **問題**: 需要科學化的超參數調整來解決過擬合
#### **解決方案**: Optuna 自動化超參數優化

```python
# yolov5c/optuna_hyperparameter_search.py
def objective(trial):
    # 自動調整學習率、正則化、損失權重等參數
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    cls_task = trial.suggest_float('cls_task', 0.1, 0.5)
    
    # 基於過擬合檢測的評分機制
    return calculate_overfitting_score()
```

#### **最佳配置結果**:
```yaml
lr0: 0.0005502189676068581
dropout: 0.20863611323495335
cls_task: 0.3828601543886939
```

---

## 📊 預期效果

### 1. **分類任務改善**
- ✅ 解決 PSAX 過擬合問題
- ✅ 平衡 PLAX 預測，防止過度偏向
- ✅ 改善整體分類性能，所有三個視圖都有合理的識別率

### 2. **訓練穩定性**
- ✅ 消除 NaN 錯誤
- ✅ 穩定訓練曲線，減少過擬合
- ✅ 獲得合理的分類準確率 (60-70%)

### 3. **醫學診斷準確性**
- ✅ 保持醫學圖像原始特徵
- ✅ 避免數據擴增對診斷的干擾
- ✅ 確保訓練和推理環境的一致性

---

## 🚀 推薦使用方案

### **方案 1: 使用 Focal Loss 配置 (推薦)**
```powershell
cd yolov5c
python train.py \
    --data ../regurgitationV1/data.yaml \
    --hyp data/hyps/hyp.classV1.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --patience 0
```

### **方案 2: 使用 Optuna 優化配置**
```powershell
cd yolov5c
python train.py \
    --data ../regurgitationV1/data.yaml \
    --hyp data/hyps/hyp.regurgitationV7.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --patience 0 \
    --name optuna_optimized_training
```

---

## ⚠️ 重要注意事項

### 1. **聯合訓練規則**
- 必須啟用分類功能，不能禁用聯合訓練
- 所有超參數文件都應配置為支持聯合訓練
- **重要：絕對不要使用 early stop** - 會影響目標檢測性能
- 關閉早停機制以獲得完整的訓練圖表和更好的檢測 mAP

### 2. **醫學圖像特殊性**
- 建議關閉數據擴增，保持原始特徵
- 避免人為干擾影響診斷結果
- 確保訓練和推理環境的一致性

### 3. **數據集快取清理**
```powershell
# 每次訓練前清理快取
$DATASET = "regurgitationV1"
$sets = @("train", "valid", "test")
foreach ($d in $sets) {
  $labels = Join-Path (Join-Path $DATASET $d) "labels"
  Remove-Item -Path (Join-Path $labels "labels.cache") -ErrorAction SilentlyContinue -Force
  Remove-Item -Path (Join-Path $labels "labels.cache.npy") -ErrorAction SilentlyContinue -Force
  Remove-Item -Path (Join-Path $labels "labels_cl.cache.npy") -ErrorAction SilentlyContinue -Force
}
```

---

## 📈 成功指標

### **檢測任務**
- mAP@0.5: > 0.65
- mAP@0.5:0.95: > 0.30
- Precision: > 0.70
- Recall: > 0.65

### **分類任務**
- 分類準確率: 60-70% (合理範圍)
- 混淆矩陣平衡: 所有三個視圖都有合理的預測分布
- 無異常的 100% 虛假準確率

### **訓練穩定性**
- 無 NaN 錯誤
- 穩定的損失曲線
- 無梯度爆炸問題

---

## 🔄 版本控制建議

1. **使用 git hard reset** 回到穩定狀態
2. **小步前進**: 任何改動都要小幅度，並能快速回退
3. **保持簡單**: 避免複雜的修改，專注於穩定性
4. **文檔記錄**: 記錄每次修改的原因和效果
5. **禁用 Early Stop**: 始終使用 `--patience 0` 以獲得最佳目標檢測性能

---

*此總結基於 `.specstory` 歷史記錄分析，涵蓋了從問題識別到解決方案實施的完整過程。*
