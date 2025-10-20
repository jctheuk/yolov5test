# Per-Class 指標提取指南

## 目前可用的指標

### ✅ 整體指標（Overall Metrics）- 已生成

**檢測指標（Detection）**：
- ✅ **Precision (P)**: 整體檢測精確度
- ✅ **Recall (R)**: 整體檢測召回率  
- ✅ **mAP@0.5**: 整體平均精度 @IoU 0.5
- ✅ **mAP@0.5:0.95**: 整體平均精度 @IoU 0.5-0.95

**分類指標（Classification）**：
- ✅ **Accuracy**: 整體分類準確率
- ✅ **Precision**: 整體分類精確度
- ✅ **Recall**: 整體分類召回率
- ✅ **F1-Score**: 整體 F1 分數

**生成的文件**：
- `results/comprehensive_metrics.csv`
- `results/comprehensive_table.tex`

### ⚠️ Per-Class 指標 - 需要運行驗證

**檢測 Per-Class（需要）**：
- ⚠️ **A4C Detection**: Precision, Recall, AP@0.5, AP@0.5:0.95
- ⚠️ **PSAX Detection**: Precision, Recall, AP@0.5, AP@0.5:0.95

**分類 Per-Class（需要）**：
- ⚠️ **A4C Classification**: Accuracy, Precision, Recall, F1-Score
- ⚠️ **PSAX Classification**: Accuracy, Precision, Recall, F1-Score

## 如何獲取 Per-Class 指標

### 選項 1：快速檢查（了解現有數據）

```powershell
python check_available_perclass_data.py
```

這會顯示：
- 哪些模型有現成的 per-class 數據
- 數據的格式和位置
- 需要提取的模型清單

### 選項 2：完整提取（推薦）

```powershell
python extract_perclass_metrics.py
```

**預期時間**：30-60 分鐘（取決於硬體）

**處理過程**：
1. 遍歷所有 12 個 YOLOv5c 模型
2. 對每個模型的 v1-v5 運行驗證
3. 提取每個類別的檢測和分類指標
4. 計算 v1-v5 的平均值
5. 生成詳細的 per-class 報告

**生成的文件**：
- `results/perclass_metrics_detailed.json` - 完整的 JSON 數據
- `results/perclass_detection_metrics.csv` - 檢測 per-class CSV
- `results/perclass_classification_metrics.csv` - 分類 per-class CSV

## 輸出文件說明

### 1. 整體指標文件（已生成）

#### comprehensive_metrics.csv
包含所有模型的整體指標：
```csv
model_type,architecture,precision,recall,mAP_0.5,mAP_0.5:0.95,cls_accuracy,cls_precision,cls_recall,cls_f1_score
YOLOv5-SC,backbone,85.1,79.8,0.795,0.349,97.4,97.5,97.4,97.5
...
```

#### comprehensive_table.tex
LaTeX 格式的綜合表格，包含：
- 檢測列：P（精確度）、R（召回率）、mAP@.5、mAP@.5:.95
- 分類列：Acc（準確率）、Prec（精確度）、Recall（召回率）、F1

### 2. Per-Class 指標文件（需要生成）

#### perclass_detection_metrics.csv
```csv
Model,Class,Precision,Recall,mAP@0.5,mAP@0.5:0.95
yolov5sc_backbone,A4C,87.2,82.1,0.812,0.356
yolov5sc_backbone,PSAX,83.0,77.5,0.778,0.342
...
```

#### perclass_classification_metrics.csv
```csv
Model,Class,Accuracy,Precision,Recall,F1-Score
yolov5sc_backbone,A4C,98.1,98.3,98.1,98.2
yolov5sc_backbone,PSAX,96.8,96.7,96.8,96.7
...
```

#### perclass_metrics_detailed.json
完整的 JSON 格式，包含所有模型、所有類別的所有指標。

## 使用範例

### 在 Python 中分析

```python
import pandas as pd

# 讀取整體指標
overall = pd.read_csv('results/comprehensive_metrics.csv')
print(overall)

# 讀取 per-class 檢測指標（運行提取後）
perclass_det = pd.read_csv('results/perclass_detection_metrics.csv')
print(perclass_det[perclass_det['Class'] == 'A4C'])

# 讀取 per-class 分類指標（運行提取後）
perclass_cls = pd.read_csv('results/perclass_classification_metrics.csv')
print(perclass_cls[perclass_cls['Class'] == 'PSAX'])
```

### 在論文中使用

```latex
% 整體比較表格
\input{results/comprehensive_table.tex}

% Per-class 檢測結果（需要自行創建表格）
% 使用 perclass_detection_metrics.csv 的數據
```

## 技術細節

### 為什麼需要運行驗證？

原始訓練結果只保存了整體指標，per-class 的詳細數據需要從模型權重重新計算：

1. **訓練時**：只記錄整體 mAP、準確率等
2. **驗證時**：可以輸出每個類別的詳細指標

### 驗證過程

```python
# 對每個模型執行
python yolov5c/val.py \
    --weights <model_weights> \
    --data <data.yaml> \
    --verbose  # 輸出 per-class 指標
```

### 指標定義

**檢測 Per-Class**：
- **Precision**: TP / (TP + FP) - 該類別檢測的精確度
- **Recall**: TP / (TP + FN) - 該類別的召回率
- **AP@0.5**: Average Precision @ IoU 0.5 - 該類別在 IoU 0.5 的 AP
- **AP@0.5:0.95**: 該類別在 IoU 0.5-0.95 的平均 AP

**分類 Per-Class**：
- **Accuracy**: 該類別的分類準確率
- **Precision**: 該類別被正確分類的比例
- **Recall**: 該類別被正確識別的比例
- **F1-Score**: Precision 和 Recall 的調和平均

## 常見問題

### Q1: 為什麼不在訓練時保存 per-class 數據？

A: 訓練腳本主要關注整體性能，per-class 數據會增加日誌大小。最佳實踐是在驗證階段提取詳細指標。

### Q2: 可以只提取部分模型的 per-class 數據嗎？

A: 可以！編輯 `extract_perclass_metrics.py` 中的 `YOLOV5C_MODELS` 列表，只保留需要的模型。

### Q3: 提取失敗了怎麼辦？

A: 檢查：
1. 模型權重文件是否存在（`best.pt` 或 `last.pt`）
2. 數據集路徑是否正確（`data.yaml`）
3. GPU 記憶體是否足夠

### Q4: 可以並行提取嗎？

A: 可以修改腳本使用多 GPU 或分批處理，但需要注意記憶體限制。

## 效能參考

基於測試環境的預估時間：

| GPU | 單個模型驗證 | 全部 60 個配置 |
|-----|-----------|-------------|
| RTX 3090 | ~1-2 分鐘 | 約 90-120 分鐘 |
| V100 | ~1-2 分鐘 | 約 90-120 分鐘 |
| T4 | ~3-4 分鐘 | 約 180-240 分鐘 |

## 下一步

1. **立即可用**：使用 `comprehensive_metrics.csv` 和 `comprehensive_table.tex` 進行整體比較
2. **獲取 Per-Class**：運行 `python extract_perclass_metrics.py`
3. **完整分析**：結合整體和 per-class 數據進行深入分析

## 快速開始

```powershell
# Step 1: 查看整體結果
python aggregate_with_perclass.py

# Step 2: 檢查可用的 per-class 數據
python check_available_perclass_data.py

# Step 3: 提取完整 per-class 指標（30-60 分鐘）
python extract_perclass_metrics.py

# Step 4: 分析結果
python -c "import pandas as pd; print(pd.read_csv('results/perclass_detection_metrics.csv'))"
```

## 相關文件

- `aggregate_with_perclass.py` - 整體指標聚合腳本
- `extract_perclass_metrics.py` - Per-class 提取腳本
- `check_available_perclass_data.py` - 數據可用性檢查
- `COMPARISON_RESULTS_SUMMARY.md` - 整體結果分析報告


