# YOLOv5 模型比較結果總結

## 生成日期
2025-10-16

## 輸出文件

本次聚合生成了以下三個文件：

1. **CSV 格式**：`results/combined_metrics.csv`
   - 包含所有模型的平均指標（v1-v5 平均值）
   - 可用於進一步數據分析和圖表生成

2. **LaTeX 表格**：`results/combined_table.tex`
   - 適合直接插入論文或報告
   - 包含完整的表格格式和標題

3. **比較圖像**：`files/1760423080004_compared@2x.jpg`
   - 視覺化比較結果
   - 包含所有模型的關鍵指標

## 模型總數

- **YOLOv5c 模型**：12 個（3 種架構 × 4 種變體）
  - YOLOv5-SC：Single Classification（backbone, p3, p4, p5）
  - YOLOv5-MC：Multi Classification（backbone, p3, p4, p5）
  - YOLOv5-MLC：Multi-Loss Classification（backbone, p3, p4, p5）

- **YOLOv5 Original 模型**：3 個
  - YOLOv5-S（Small）
  - YOLOv5-M（Medium）
  - YOLOv5-L（Large）

**總計**：15 個模型

## 主要發現

### 檢測性能（mAP）

#### mAP@0.5 排名
1. **YOLOv5-SC backbone**: 0.795 （最佳）
2. **YOLOv5-SC p5**: 0.777
3. **YOLOv5-SC p4**: 0.766
4. **YOLOv5-SC p3**: 0.765
5. **YOLOv5-MC/MLC backbone**: ~0.749

#### mAP@0.5:0.95 排名
1. **YOLOv5-SC backbone**: 0.349 （最佳）
2. **YOLOv5-SC p5**: 0.346
3. **YOLOv5-SC p3**: 0.339
4. **YOLOv5-SC p4**: 0.335
5. **YOLOv5-MC backbone**: 0.298

**結論**：YOLOv5-SC（Single Classification）在檢測任務上表現最佳，其中 backbone 架構達到最高 mAP。

### 分類性能（Accuracy）

#### 分類準確率排名
1. **YOLOv5-M（Original）**: 98.25% （最佳）
2. **YOLOv5-L（Original）**: 97.37%
3. **YOLOv5-S（Original）**: 97.78%
4. **YOLOv5-SC backbone**: 97.45%
5. **YOLOv5-SC p5**: 97.45%
6. **YOLOv5-MC p3**: 97.34%

**結論**：YOLOv5 Original 模型（純分類）在分類任務上表現最佳，其中 Medium 模型達到 98.25% 的準確率。

### 聯合訓練模型比較

在 YOLOv5c 聯合訓練模型中：

1. **YOLOv5-SC（Single Classification）**：
   - 檢測性能最佳（mAP@0.5: 0.795）
   - 分類性能良好（Accuracy: 97.45%）
   - **推薦用於需要平衡檢測和分類的場景**

2. **YOLOv5-MC（Multi Classification）**：
   - 檢測性能中等（mAP@0.5: 0.749）
   - 分類性能優秀（Accuracy: 97.34%）
   - 適合需要多類別分類的場景

3. **YOLOv5-MLC（Multi-Loss Classification）**：
   - 檢測性能中等（mAP@0.5: 0.749）
   - 分類性能良好（Accuracy: 97.23%）
   - 使用多損失策略，適合複雜場景

### 架構比較（YOLOv5c）

在不同的特徵提取架構中：

1. **Backbone**：
   - 檢測性能最佳
   - 分類性能穩定
   - **推薦作為首選架構**

2. **P5**：
   - 檢測性能良好
   - 分類性能穩定
   - 適合需要多尺度特徵的場景

3. **P3**：
   - 檢測性能中等
   - 分類性能良好
   - 適合小目標檢測

4. **P4**：
   - 檢測性能中等
   - 分類性能略低
   - 平衡的選擇

## 數據來源

- **YOLOv5c 數據**：來自 `thesis_results_complete.xlsx`
  - 60 筆記錄（12 個模型 × 5 個版本）
  - 包含檢測和分類的完整指標

- **YOLOv5 Original 數據**：來自 `YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md`
  - 15 筆記錄（3 個模型 × 5 個版本）
  - 僅包含分類準確率

## 聚合方法

- 對每個模型的 v1-v5 版本取平均值
- 檢測指標：mAP@0.5 和 mAP@0.5:0.95
- 分類指標：Accuracy、Precision、Recall、F1-Score

## 使用說明

### 在 LaTeX 中使用

直接將 `results/combined_table.tex` 插入到您的論文中：

```latex
\input{results/combined_table.tex}
```

### 在 Excel/Python 中分析

讀取 `results/combined_metrics.csv` 進行進一步分析：

```python
import pandas as pd

df = pd.read_csv('results/combined_metrics.csv')
print(df)
```

### 在簡報中使用

直接使用 `files/1760423080004_compared@2x.jpg` 圖像文件。

## 建議

基於本次比較結果，我們提出以下建議：

1. **純分類任務**：使用 YOLOv5-M（Original），準確率最高（98.25%）

2. **純檢測任務**：使用 YOLOv5-SC backbone，mAP 最高（0.795）

3. **聯合任務**：
   - 優先考慮 **YOLOv5-SC backbone**：檢測和分類性能都很好
   - 需要多類別：使用 **YOLOv5-MC p3**：分類性能最佳的聯合模型

4. **資源受限場景**：使用 YOLOv5-SC p3 或 p4，性能和效率的良好平衡

## 腳本文件

聚合腳本：`aggregate_thesis_results.py`

主要功能：
- 從 Excel 讀取論文結果
- 按模型類型和架構聚合
- 生成 CSV、LaTeX 和圖像輸出

運行命令：
```powershell
python aggregate_thesis_results.py
```

## 相關文件

- 原始數據：`thesis_results_complete.xlsx`
- 分析報告：`YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md`
- 聚合腳本：`aggregate_thesis_results.py`
- 舊版腳本：`aggregate_yolov5c_metrics.py`（已棄用）


