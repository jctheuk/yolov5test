# YOLOv5C - 聯合檢測與分類系統說明

## 概述

YOLOv5C 是一個基於 YOLOv5 的增強版本，專門設計用於**同時進行目標檢測和圖像分類**的聯合訓練系統。這個系統特別針對醫學圖像分析（如心臟超音波圖像）進行了優化，能夠在檢測特定病變的同時，對圖像的視角進行分類。

## 🎯 核心功能

### 1. 雙任務架構 (Dual-Task Architecture)
- **目標檢測**: 檢測圖像中的特定對象（如心臟瓣膜病變：AR、MR、PR、TR）
- **圖像分類**: 對圖像視角進行分類（如心臟超音波視角：PSAX、PLAX、A4C）
- **聯合訓練**: 兩個任務同時進行，共享特徵提取器，提高整體性能

### 2. 增強的分類頭 (Enhanced Classification Head)
```python
class YOLOv5WithClassification(nn.Module):
    def __init__(self, in_channels, num_classes):
        # 特徵提取層
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
```

### 3. 智能損失函數 (Intelligent Loss Function)
```python
class ComputeLoss:
    def __init__(self, model, autobalance=False):
        # 檢測損失
        self.BCEcls = nn.BCEWithLogitsLoss()  # 分類損失
        self.BCEobj = nn.BCEWithLogitsLoss()  # 目標性損失
        
        # 分類任務損失
        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.NLLLoss()
        self.cls_task_loss_weight = 0.3  # 分類任務權重
```

## 🏗️ 系統架構

### 模型結構
```
輸入圖像 (640x640x3)
    ↓
YOLOv5 骨幹網路 (Backbone)
    ↓
特徵金字塔網路 (FPN)
    ↓
    ├── 檢測頭 (Detection Head) → 目標檢測結果
    └── 分類頭 (Classification Head) → 圖像分類結果
```

### 配置文件範例 (yolov5sc.yaml)
```yaml
# 參數配置
nc: 4  # 檢測類別數：AR, MR, PR, TR
num_cls: 3  # 分類類別數：PSAX, PLAX, A4C

# 模型架構
head:
  # 檢測頭
  [[17, 20, 23], 1, Detect, [nc, anchors]]
  # 分類頭
  [17, 1, YOLOv5WithClassification, [128, 3]]
```

## 📊 訓練流程

### 1. 數據準備
```yaml
# data.yaml 配置
path: ../path/to/dataset
train: images/train
val: images/val

# 檢測類別
nc: 4
names: ['AR', 'MR', 'PR', 'TR']

# 分類類別
num_cls: 3
cls_names: ['PSAX', 'PLAX', 'A4C']
```

### 2. 標籤格式
每個標籤文件包含兩行：
```
# 第一行：檢測標籤
class_id x_center y_center width height

# 第二行：分類標籤
class_index
```

範例：
```
0 0.5 0.5 0.2 0.3  # 檢測：AR 類別，位置 (0.5, 0.5)，大小 0.2x0.3
1                   # 分類：PLAX 視角
```

### 3. 訓練命令
```bash
python train.py \
    --data data.yaml \
    --cfg models/yolov5sc.yaml \
    --weights yolov5s.pt \
    --epochs 50 \
    --batch-size 16 \
    --device auto
```

## 🔧 關鍵改進

### 1. 特徵提取優化
- **多層卷積**: 使用多層卷積進行特徵提取
- **批標準化**: 每層都使用 BatchNorm 提高訓練穩定性
- **激活函數**: 使用 SiLU 激活函數提高性能
- **權重初始化**: 使用 Kaiming 初始化提高收斂速度

### 2. 損失函數平衡
- **動態權重**: 檢測和分類損失的動態平衡
- **標籤平滑**: 提高分類任務的泛化能力
- **梯度裁剪**: 防止梯度爆炸
- **溫度調節**: 控制 Softmax 的銳度

### 3. 數據處理增強
- **雙標籤支持**: 同時處理檢測和分類標籤
- **批次處理**: 高效的批次數據處理
- **錯誤處理**: 對格式錯誤的標籤進行健壯處理

## 📈 性能指標

### 訓練監控
- **檢測損失**: Box loss, Object loss, Class loss
- **分類損失**: Cross-entropy loss
- **分類準確率**: 每個 epoch 的準確率
- **驗證指標**: 兩個任務的獨立驗證指標

### 輸出文件
- `results.png`: 訓練曲線圖
- `classification_metrics.png`: 分類性能圖表
- `best.pt`: 最佳模型權重
- `last.pt`: 最新模型權重

## 🎛️ 超參數配置

### 關鍵超參數
```yaml
# 損失係數
box: 0.05          # 邊框損失權重
cls: 0.5           # 分類損失權重
obj: 1.0           # 目標性損失權重
cls_task: 0.3      # 分類任務損失權重

# 分類特定參數
label_smoothing: 0.1  # 標籤平滑
temperature: 1.0      # Softmax 溫度
```

## 🔍 使用場景

### 醫學圖像分析
- **心臟超音波**: 檢測瓣膜病變 + 視角分類
- **X光片分析**: 檢測病變 + 身體部位分類
- **CT掃描**: 檢測腫瘤 + 器官分類

### 其他應用
- **自動駕駛**: 檢測車輛 + 道路類型分類
- **安防監控**: 檢測人員 + 行為分類
- **工業檢測**: 檢測缺陷 + 產品類型分類

## 🚀 優勢特點

### 1. 效率提升
- **共享特徵**: 兩個任務共享特徵提取器，減少計算量
- **端到端訓練**: 一次性訓練完成兩個任務
- **推理加速**: 單次前向傳播獲得兩個結果

### 2. 性能提升
- **特徵互補**: 檢測和分類任務相互促進
- **數據利用**: 充分利用標註數據
- **泛化能力**: 聯合訓練提高模型泛化能力

### 3. 實用性
- **醫學應用**: 特別適合醫學圖像分析
- **易於部署**: 基於成熟的 YOLOv5 架構
- **可擴展性**: 容易添加新的檢測或分類類別

## 📝 注意事項

### 訓練建議
1. **數據質量**: 確保高質量的標註數據
2. **批次大小**: 根據 GPU 記憶體調整批次大小
3. **學習率**: 聯合訓練建議使用較低的學習率
4. **早停機制**: 建議關閉早停以獲得完整訓練曲線

### 常見問題
1. **分類性能差**: 檢查標籤格式和類別數量
2. **訓練不穩定**: 降低學習率或增加批次大小
3. **記憶體不足**: 減少批次大小或圖像尺寸

## 🔄 與標準 YOLOv5 的差異

| 特性 | 標準 YOLOv5 | YOLOv5C |
|------|-------------|---------|
| 任務類型 | 僅檢測 | 檢測 + 分類 |
| 模型輸出 | 檢測結果 | 檢測 + 分類結果 |
| 損失函數 | 檢測損失 | 檢測 + 分類損失 |
| 標籤格式 | 檢測標籤 | 檢測 + 分類標籤 |
| 應用場景 | 通用目標檢測 | 醫學圖像分析 |

## 📚 技術細節

### 模型初始化
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
```

### 前向傳播
```python
def forward(self, x):
    # 特徵提取
    features = self.feature_extractor(x)
    
    # 全局池化
    pooled = self.avgpool(features)
    
    # 展平
    flattened = self.flatten(pooled)
    
    # 分類
    output = self.classifier(flattened)
    
    return output
```

## 🎯 總結

YOLOv5C 是一個強大的聯合檢測與分類系統，特別適合需要同時進行目標檢測和圖像分類的應用場景。通過共享特徵提取器和智能損失函數設計，它能夠在保持高效推理速度的同時，提供優異的檢測和分類性能。對於醫學圖像分析等專業領域，YOLOv5C 提供了一個完整且易用的解決方案。
