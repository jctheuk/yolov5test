# YOLOv5 分類任務整合指南

## 概述

本指南說明如何將成功的分類任務實現整合到 YOLOv5 的核心 `loss.py` 和 `train.py` 文件中。

---

## 當前成功的實現

### 文件結構
- **損失計算**：`yolov5c/utils/classification_task_loss.py`
- **訓練腳本**：`train_classification_task.py`

### 核心特性
✅ 手動 CrossEntropy 實現（避免 PyTorch 版本問題）  
✅ One-hot 編碼標籤處理  
✅ 聯合檢測和分類訓練  
✅ 兼容 PyTorch 21.08 和更新版本  

---

## 整合步驟

### 第一步：修改 `utils/loss.py`

#### 1.1 添加手動 CrossEntropy 實現

在 `ComputeLoss` 類中添加以下方法：

```python
def manual_cross_entropy_loss(self, logits, targets):
    """
    手動 CrossEntropy 損失實現，用於最大兼容性
    等效於 nn.CrossEntropyLoss() 但適用於所有 PyTorch 版本
    
    Args:
        logits: 模型預測 [batch_size, num_classes]
        targets: 目標類別索引 [batch_size]
        
    Returns:
        CrossEntropy 損失值
    """
    import torch.nn.functional as F
    
    # 計算 log softmax
    log_probs = F.log_softmax(logits, dim=1)
    
    # 收集目標類別的 log 概率
    batch_size = logits.shape[0]
    target_log_probs = log_probs[range(batch_size), targets]
    
    # 返回負對數似然（CrossEntropy 損失）
    return -target_log_probs.mean()
```

#### 1.2 修改 `__init__` 方法

```python
def __init__(self, model, autobalance=False):
    self.sort_obj_iou = False
    self.balance = {3: [4.0, 1.0, 0.4]}
    self.ssi = list(self.balance.keys())[0]
    self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
    self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
    self.gr = 1.0
    self.autobalance = autobalance
    
    # 添加分類任務支持
    self.enable_classification = h.get('classification_enabled', False)
    self.cls_task_weight = h.get('cls_task', 0.3)
    
    # 使用手動實現避免 PyTorch 版本問題
    self.classification_criterion = None  # 使用手動實現
```

#### 1.3 修改 `__call__` 方法

在 `__call__` 方法中添加分類損失計算：

```python
def __call__(self, p, targets, cls_targets=None):  # predictions, targets, classification_targets
    device = targets.device
    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)
    lcls_task = torch.zeros(1, device=device)  # 添加分類任務損失
    
    # ... 原有的檢測損失計算 ...
    
    # 分類任務損失計算
    if self.enable_classification and len(p) == 2:
        detection_outputs, classification_output = p
        
        if classification_output is not None and cls_targets is not None:
            # 確保標籤在同一設備上
            cls_targets = cls_targets.to(classification_output.device)
            
            # 處理 one-hot 編碼和類別索引格式
            if cls_targets.dim() > 1 and cls_targets.shape[-1] > 1:
                # One-hot 編碼: [batch_size, num_classes] -> [batch_size]
                target_indices = cls_targets.argmax(dim=-1).long()
            elif cls_targets.dim() > 1:
                # 帶有額外維度的類別索引: [batch_size, 1] -> [batch_size]
                target_indices = cls_targets.squeeze().long()
            else:
                # 已經是 1D 類別索引
                target_indices = cls_targets.long()
            
            num_classes = classification_output.shape[-1]
            
            # 確保目標在有效範圍內
            if target_indices.max() >= num_classes:
                target_indices = torch.clamp(target_indices, 0, num_classes - 1)
            
            # 使用手動 CrossEntropy 實現
            lcls_task = self.manual_cross_entropy_loss(classification_output, target_indices)
    
    # 計算總損失
    loss = lbox * self.hyp['box'] + lobj * self.hyp['obj'] + lcls * self.hyp['cls']
    
    # 添加分類任務損失
    if self.enable_classification:
        loss += lcls_task * self.cls_task_weight
    
    return loss * bs, torch.cat((lbox, lobj, lcls, lcls_task)).detach()
```

---

### 第二步：修改 `train.py`

#### 2.1 添加分類標籤處理

在訓練循環中，處理分類標籤：

```python
for i, (imgs, targets, paths, _) in pbar:  # batch
    ni = i + nb * epoch
    imgs = imgs.to(device, non_blocking=True).float() / 255
    
    # 提取分類標籤（如果有的話）
    classification_labels = None
    if hasattr(dataset, 'classification_labels'):
        # 從數據集獲取分類標籤
        classification_labels = dataset.get_classification_labels(batch_indices)
        classification_labels = classification_labels.to(device)
        
        # 處理 one-hot 編碼
        if classification_labels.dim() > 1 and classification_labels.shape[-1] > 1:
            # 已經是正確格式，無需處理
            pass
        
        # 確保是 long 類型
        if classification_labels.dtype != torch.long:
            if classification_labels.dim() > 1:
                # One-hot 編碼，轉換為索引
                classification_labels = classification_labels.argmax(dim=-1)
            classification_labels = classification_labels.long()
```

#### 2.2 修改前向傳播

```python
# Forward
with amp.autocast(enabled=cuda):
    pred = model(imgs)  # forward
    
    # 計算損失
    if hasattr(model, 'classification_head') and classification_labels is not None:
        # 聯合訓練模式：檢測 + 分類
        loss, loss_items = compute_loss((pred, model.classification_output), targets, classification_labels)
    else:
        # 純檢測模式
        loss, loss_items = compute_loss(pred, targets)
```

#### 2.3 更新超參數文件

在 `data/hyps/hyp.scratch-low.yaml` 添加：

```yaml
# Classification task settings
classification_enabled: True  # 啟用分類任務
cls_task: 0.3  # 分類任務損失權重
label_smoothing: 0.0  # Label smoothing（設為 0 以避免版本問題）
```

---

### 第三步：修改模型架構

#### 3.1 在 `models/yolo.py` 中添加分類頭

```python
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        # ... 原有初始化 ...
        
        # 添加分類頭（如果啟用）
        if self.hyp.get('classification_enabled', False):
            self.classification_head = self._create_classification_head()
            self.classification_output = None
    
    def _create_classification_head(self):
        """創建分類頭"""
        # 使用最後一層特徵
        in_channels = 1280  # 根據您的模型調整
        num_classes = self.nc  # 使用相同的類別數
        
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
    
    def forward(self, x):
        # 原有的檢測前向傳播
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        
        # 添加分類頭
        if hasattr(self, 'classification_head'):
            # 使用最後一層特徵進行分類
            self.classification_output = self.classification_head(y[-1])
            return (x, self.classification_output)
        
        return x
```

---

## 關鍵差異說明

### 與原始 YOLOv5 Classify 的差異

| 特性 | YOLOv5 Classify | 本實現 |
|------|----------------|--------|
| **任務類型** | 純分類 | 聯合檢測+分類 |
| **標籤格式** | 類別索引 `[batch_size]` | One-hot 編碼 `[batch_size, num_classes]` |
| **損失函數** | `nn.CrossEntropyLoss()` | 手動實現（兼容性更好） |
| **模型輸出** | 單一分類輸出 | 檢測輸出 + 分類輸出 |
| **版本兼容** | 依賴 PyTorch 版本 | 完全兼容所有版本 |

### 為什麼需要手動實現

1. **PyTorch 版本問題**：不同版本的 `nn.CrossEntropyLoss` 對 label smoothing 的處理不同
2. **One-hot 編碼支持**：自動處理 one-hot 編碼標籤
3. **設備兼容性**：避免 CUDA 和 CPU 之間的類型不匹配

---

## 驗證清單

整合完成後，請驗證以下項目：

- [ ] 訓練可以正常啟動
- [ ] 損失值正常（不是 NaN 或 Inf）
- [ ] 分類準確率正常顯示
- [ ] 檢測和分類都有輸出
- [ ] 驗證階段正常運行
- [ ] 模型可以正常保存和加載
- [ ] 推理時同時輸出檢測和分類結果

---

## 故障排除

### 問題 1：RuntimeError: Expected floating point type

**解決方案**：確保使用手動 CrossEntropy 實現，而不是 `nn.CrossEntropyLoss()`

### 問題 2：形狀不匹配錯誤

**解決方案**：檢查 one-hot 編碼處理邏輯，確保正確轉換為類別索引

### 問題 3：標籤設備不匹配

**解決方案**：確保在損失計算前將標籤移到正確設備：
```python
cls_targets = cls_targets.to(classification_output.device)
```

---

## 參考文件

- **當前成功實現**：`yolov5c/utils/classification_task_loss.py`
- **訓練腳本**：`train_classification_task.py`
- **原始 YOLOv5 Classify**：`yolov5original/classify/train.py`

---

## 總結

本整合方案提供了一個穩健的聯合訓練解決方案，結合了：

1. ✅ **手動 CrossEntropy 實現** - 避免所有 PyTorch 版本問題
2. ✅ **One-hot 編碼處理** - 自動轉換為類別索引
3. ✅ **聯合訓練支持** - 同時訓練檢測和分類
4. ✅ **完全向後兼容** - 支持 PyTorch 21.08 及更新版本

如果訓練結果良好，按照本指南進行整合即可將成功的實現遷移到核心文件中。
