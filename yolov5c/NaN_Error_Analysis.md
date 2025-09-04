# YOLOv5 聯合訓練 NaN 錯誤分析與解決方案

## 概述

在 YOLOv5 聯合檢測和分類訓練中，NaN（Not a Number）錯誤是一個常見但嚴重的問題。本文檔詳細分析了可能的原因和解決方案。

## 什麼是 NaN 錯誤？

NaN 錯誤發生在深度學習訓練過程中，當數值計算產生無效結果時：
- **梯度爆炸**: 梯度值變得極大
- **梯度消失**: 梯度值接近零
- **數值不穩定**: 損失函數產生無效值

## 在您的案例中可能的原因

### 1. **聯合訓練的複雜性**

#### 問題描述
```python
# 您的模型同時執行兩個任務
- 檢測任務: 4個類別 (AR, MR, PR, TR)
- 分類任務: 3個類別 (PSAX, PLAX, A4C)
```

#### 為什麼會導致 NaN
- **損失權重不平衡**: 兩個任務的損失可能相差幾個數量級
- **梯度衝突**: 檢測和分類的梯度可能相互抵消
- **學習率不匹配**: 不同任務需要不同的學習率

### 2. **超參數設定問題**

#### 原始設定分析
```yaml
# 您的原始超參數
lr0: 0.01                    # 學習率可能過高
cls_task: 0.517             # 分類權重可能過高
batch-size: 128             # 批次大小可能過大
warmup_epochs: 10.0         # 暖身期可能不夠
```

#### 問題分析
- **學習率 0.01**: 對於聯合訓練來說太高
- **批次大小 128**: 可能導致梯度不穩定
- **分類權重 0.517**: 可能與檢測損失不平衡

### 3. **數據相關問題**

#### 醫學圖像特殊性
```python
# 醫學圖像的特點
- 高對比度
- 複雜的解剖結構
- 標註不一致
- 類別不平衡
```

#### 可能影響
- **標註錯誤**: 導致損失計算異常
- **類別不平衡**: 某些類別樣本太少
- **圖像品質**: 極端亮度或對比度

### 4. **模型架構問題**

#### 聯合訓練架構
```yaml
# 您的模型架構
backbone: YOLOv5 backbone
detection_head: Detect layer (4 classes)
classification_head: YOLOv5WithClassification (3 classes)
```

#### 潛在問題
- **特徵共享**: 兩個頭可能競爭特徵
- **權重初始化**: 新增的分類層初始化不當
- **梯度流動**: 梯度在兩個頭之間分配不均

## 診斷方法

### 1. **檢查訓練日誌**
```bash
# 查看訓練過程中的異常
grep -i "nan\|inf" training.log
grep -i "loss" training.log | tail -20
```

### 2. **監控關鍵指標**
```python
# 需要監控的指標
- 檢測損失 (box_loss, obj_loss, cls_loss)
- 分類損失 (cls_task_loss)
- 總損失 (total_loss)
- 梯度範數 (gradient norm)
- 學習率變化
```

### 3. **數據檢查**
```python
# 檢查數據集
- 標註文件是否完整
- 圖像是否損壞
- 類別分布是否平衡
- 標註格式是否正確
```

## 解決方案

### 方案 1: 調整超參數（推薦）

#### 修改 `hyp.fixed_classification.yaml`
```yaml
# 降低學習率
lr0: 0.001  # 從 0.01 降低到 0.001

# 減少分類權重
cls_task: 0.3  # 從 0.517 降低到 0.3

# 調整損失權重
cls: 0.3  # 從 0.5 降低到 0.3
obj: 0.7  # 從 1.0 降低到 0.7

# 增加暖身期
warmup_epochs: 15.0  # 從 10.0 增加到 15.0

# 減少數據擴增
hsv_s: 0.2  # 從 0.4 降低到 0.2
hsv_v: 0.2  # 從 0.3 降低到 0.2
degrees: 5.0  # 從 10.0 降低到 5.0
```

#### 修改訓練命令
```bash
# 減少批次大小
--batch-size 64  # 從 128 降低到 64

# 減少工作進程
--workers 4  # 從默認值降低到 4

# 關閉早停
--patience 0  # 確保完整訓練
```

### 方案 2: 增強代碼穩定性

#### 添加 NaN 檢測
```python
# 在 train.py 中添加
if torch.isnan(total_loss) or torch.isinf(total_loss):
    LOGGER.warning(f"NaN/Inf loss detected: {total_loss.item()}")
    continue

# 增強梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

#### 改進損失計算
```python
# 確保損失計算的數值穩定性
total_loss = torch.clamp(total_loss, min=1e-8, max=1e8)
```

### 方案 3: 數據預處理

#### 清理數據集
```bash
# 清理快取文件
rm -rf ../Regurgitation-YOLODataset-Detection/*/labels/*.cache*

# 驗證標註
python utils/check_labels.py --data ../Regurgitation-YOLODataset-Detection/data.yaml
```

#### 平衡數據集
```python
# 檢查類別分布
import pandas as pd
import matplotlib.pyplot as plt

# 分析每個類別的樣本數量
# 如果嚴重不平衡，考慮數據增強或重採樣
```

### 方案 4: 模型架構調整

#### 分階段訓練
```bash
# 第一階段：只訓練檢測
python train.py --cfg models/yolov5s.yaml --data data.yaml --epochs 100

# 第二階段：凍結檢測層，訓練分類層
python train.py --cfg models/yolov5sc.yaml --data data.yaml --epochs 50 --freeze 0-17

# 第三階段：聯合微調
python train.py --cfg models/yolov5sc.yaml --data data.yaml --epochs 150
```

#### 調整分類頭
```yaml
# 在 yolov5sc.yaml 中調整
[17, 1, YOLOv5WithClassification, [256, 3]]  # 減少通道數
```

## 預防措施

### 1. **漸進式訓練**
- 從小批次開始
- 逐步增加批次大小
- 監控損失變化

### 2. **定期檢查點**
```python
# 每 N 個 epoch 保存檢查點
if epoch % 10 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

### 3. **監控工具**
```python
# 使用 TensorBoard 監控
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/total', total_loss.item(), global_step)
```

## 緊急處理

### 如果遇到 NaN 錯誤

#### 立即停止訓練
```bash
# 按 Ctrl+C 停止訓練
# 保存最後的檢查點
```

#### 分析錯誤
```bash
# 檢查日誌文件
tail -50 training.log

# 檢查 GPU 記憶體
nvidia-smi
```

#### 恢復訓練
```bash
# 使用較安全的參數重新開始
python train.py --hyp data/hyps/hyp.nan_fix.yaml --batch-size 32
```

## 總結

NaN 錯誤在聯合訓練中很常見，主要原因是：
1. **學習率過高**
2. **批次大小過大**
3. **損失權重不平衡**
4. **數據問題**

**推薦解決順序**：
1. 降低學習率到 0.001
2. 減少批次大小到 64
3. 調整分類權重到 0.3
4. 增加暖身期到 15 epochs
5. 清理數據快取
6. 如果仍有問題，考慮分階段訓練

通過這些調整，您的聯合訓練應該能夠穩定進行，避免 NaN 錯誤。
