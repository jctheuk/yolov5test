# YOLOv5 模型比較結果聚合工具

## 快速開始

### 運行聚合腳本

使用 PowerShell：
```powershell
.\run_aggregate_results.ps1
```

或直接使用 Python：
```powershell
python aggregate_thesis_results.py
```

## 輸出文件

運行後會生成以下三個文件：

### 1. CSV 格式數據
- **文件**：`results/combined_metrics.csv`
- **用途**：數據分析、圖表生成、Excel 處理
- **內容**：所有模型的平均指標（v1-v5）

### 2. LaTeX 表格
- **文件**：`results/combined_table.tex`
- **用途**：直接插入論文或報告
- **使用方式**：
  ```latex
  \input{results/combined_table.tex}
  ```

### 3. 比較圖像
- **文件**：`files/1760423080004_compared@2x.jpg`
- **用途**：簡報、海報、論文圖表
- **格式**：高清 JPEG 圖像

## 數據來源

### YOLOv5c 模型（12 個）
- **來源**：`thesis_results_complete.xlsx`
- **包含**：
  - 檢測指標：mAP@0.5、mAP@0.5:0.95
  - 分類指標：Accuracy、Precision、Recall、F1-Score
- **模型**：
  - YOLOv5-SC（Single Classification）× 4 架構
  - YOLOv5-MC（Multi Classification）× 4 架構
  - YOLOv5-MLC（Multi-Loss Classification）× 4 架構

### YOLOv5 Original 模型（3 個）
- **來源**：`YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md`
- **包含**：
  - 分類指標：整體準確率
- **模型**：
  - YOLOv5-S（Small）
  - YOLOv5-M（Medium）
  - YOLOv5-L（Large）

## 聚合方法

對每個模型的 v1-v5 數據集版本取平均值：

```
平均 mAP@0.5 = (v1_mAP + v2_mAP + v3_mAP + v4_mAP + v5_mAP) / 5
平均準確率 = (v1_acc + v2_acc + v3_acc + v4_acc + v5_acc) / 5
```

## 主要發現

### 🏆 最佳檢測模型
**YOLOv5-SC backbone**
- mAP@0.5: 0.795
- mAP@0.5:0.95: 0.349
- 分類準確率: 97.45%

### 🏆 最佳分類模型
**YOLOv5-M（Original）**
- 分類準確率: 98.25%
- 僅分類任務（無檢測）

### 🏆 最佳聯合模型
**YOLOv5-SC backbone**
- 檢測和分類性能均衡
- 推薦用於需要同時進行檢測和分類的場景

## 詳細分析

請參閱 **[COMPARISON_RESULTS_SUMMARY.md](COMPARISON_RESULTS_SUMMARY.md)** 獲取：
- 完整的模型排名
- 詳細的性能比較
- 使用建議
- 技術分析

## 文件結構

```
yolov5test/
├── aggregate_thesis_results.py    # 主聚合腳本
├── run_aggregate_results.ps1      # PowerShell 運行腳本
├── AGGREGATE_RESULTS_README.md    # 本文件
├── COMPARISON_RESULTS_SUMMARY.md  # 詳細分析報告
├── thesis_results_complete.xlsx   # 原始數據
├── results/
│   ├── combined_metrics.csv       # 輸出：CSV 數據
│   └── combined_table.tex         # 輸出：LaTeX 表格
└── files/
    └── 1760423080004_compared@2x.jpg  # 輸出：比較圖像
```

## 腳本說明

### aggregate_thesis_results.py

主要功能：
1. 讀取 `thesis_results_complete.xlsx`
2. 按模型類型和架構分組
3. 計算 v1-v5 的平均值
4. 生成三種格式的輸出

主要函數：
- `load_thesis_results()`: 載入 Excel 數據
- `aggregate_by_model()`: 按模型聚合
- `save_to_csv()`: 生成 CSV
- `save_to_latex()`: 生成 LaTeX 表格
- `create_comparison_image()`: 生成比較圖像

### run_aggregate_results.ps1

簡化的 PowerShell 腳本：
- 自動運行聚合腳本
- 顯示生成的文件路徑
- 提供錯誤處理

## 依賴項

Python 套件：
```
pandas
openpyxl
numpy
pillow
```

安裝：
```powershell
pip install pandas openpyxl numpy pillow
```

## 常見問題

### Q: 如何重新生成結果？
A: 直接運行腳本即可，會覆蓋舊文件：
```powershell
.\run_aggregate_results.ps1
```

### Q: 可以修改數據來源嗎？
A: 可以，編輯 `aggregate_thesis_results.py` 中的以下部分：
```python
excel_path = os.path.join(base_path, 'thesis_results_complete.xlsx')
```

### Q: 圖像沒有生成？
A: 檢查是否存在基礎圖像：`files/1760423080004@2x.jpg`
如果不存在，腳本會創建一個白色背景的圖像。

### Q: LaTeX 表格如何調整？
A: 編輯 `save_to_latex()` 函數中的表格格式：
```python
latex_lines.append(r"\begin{tabular}{l|l|cc|cccc}")
```

## 更新日誌

### 2025-10-16
- ✅ 創建初始版本
- ✅ 實現從 Excel 讀取數據
- ✅ 添加 YOLOv5 Original 結果
- ✅ 生成三種格式輸出
- ✅ 修復百分比格式問題
- ✅ 創建 PowerShell 運行腳本
- ✅ 添加詳細文檔

## 聯繫資訊

如有問題或建議，請查看：
- 主要分析報告：`COMPARISON_RESULTS_SUMMARY.md`
- 原始訓練分析：`YOLOV5ORIGINAL_CLASSIFICATION_TRAINING_ANALYSIS.md`
- 論文結果數據：`thesis_results_complete.xlsx`


