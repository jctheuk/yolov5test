# 當前狀態與下一步

## ✅ 已完成的工作

### 1. 整體指標聚合（Overall Metrics）

已生成包含**整體檢測和分類指標**的完整比較：

#### 檢測指標（Detection - Overall）
- ✅ **Precision（P）**: 整體檢測精確度
- ✅ **Recall（R）**: 整體檢測召回率
- ✅ **mAP@0.5**: 整體 IoU 0.5 的平均精度
- ✅ **mAP@0.5:0.95**: 整體 IoU 0.5-0.95 的平均精度

#### 分類指標（Classification - Overall）
- ✅ **Accuracy**: 整體分類準確率
- ✅ **Precision**: 整體分類精確度
- ✅ **Recall**: 整體分類召回率
- ✅ **F1-Score**: 整體 F1 分數

#### 生成的文件
1. **`results/comprehensive_metrics.csv`**
   - 15 個模型的完整整體指標
   - CSV 格式，易於分析

2. **`results/comprehensive_table.tex`**
   - LaTeX 格式表格
   - 包含檢測和分類的所有整體指標
   - 可直接插入論文

3. **`results/combined_metrics.csv`** 和 **`results/combined_table.tex`**
   - 簡化版本（不含 P/R）
   - 向後兼容

4. **`files/1760423080004_compared@2x.jpg`**
   - 視覺化比較圖像

### 2. 分析報告

- **`COMPARISON_RESULTS_SUMMARY.md`** - 詳細的整體分析
- **`AGGREGATE_RESULTS_README.md`** - 使用說明
- **`TASK_COMPLETION_REPORT.md`** - 完成報告

## ⚠️ 需要額外提取的數據

### Per-Class 指標（需要運行驗證）

您要求的 **per-class** 數據目前還沒有，需要運行驗證來提取：

#### 檢測 Per-Class（A4C, PSAX 等）
- ⚠️ A4C - Precision, Recall, AP@0.5, AP@0.5:0.95
- ⚠️ PSAX - Precision, Recall, AP@0.5, AP@0.5:0.95

#### 分類 Per-Class（A4C, PSAX 等）
- ⚠️ A4C - Accuracy, Precision, Recall, F1-Score
- ⚠️ PSAX - Accuracy, Precision, Recall, F1-Score

## 📋 如何獲取 Per-Class 數據

### 方案 1：快速檢查現有數據（1 分鐘）

```powershell
python check_available_perclass_data.py
```

這會告訴您：
- 哪些驗證結果已經有 per-class 數據
- 數據的位置和格式
- 需要提取的模型清單

### 方案 2：完整提取 Per-Class 指標（30-60 分鐘）⭐ 推薦

```powershell
.\run_perclass_extraction.ps1
```

或直接運行：
```powershell
python extract_perclass_metrics.py
```

**這會做什麼？**
1. 遍歷所有 12 個 YOLOv5c 模型
2. 對每個模型的 v1-v5 運行驗證
3. 從驗證輸出提取 per-class 指標
4. 計算 v1-v5 的平均值
5. 生成三個輸出文件：
   - `results/perclass_metrics_detailed.json` - 完整數據
   - `results/perclass_detection_metrics.csv` - 檢測 per-class
   - `results/perclass_classification_metrics.csv` - 分類 per-class

**預期時間**：
- GPU (V100/3090): 約 30-60 分鐘
- GPU (T4): 約 60-120 分鐘

## 📊 輸出文件對照表

### 當前可用的文件（整體指標）

| 文件 | 內容 | 用途 |
|------|------|------|
| `results/comprehensive_metrics.csv` | 15 個模型的整體檢測+分類指標 | 數據分析、Excel |
| `results/comprehensive_table.tex` | LaTeX 格式的綜合表格 | 論文撰寫 |
| `files/1760423080004_compared@2x.jpg` | 視覺化比較圖 | 簡報、海報 |

### 運行提取後會有的文件（Per-Class 指標）

| 文件 | 內容 | 用途 |
|------|------|------|
| `results/perclass_detection_metrics.csv` | 每個模型+每個類別的檢測指標 | Per-class 檢測分析 |
| `results/perclass_classification_metrics.csv` | 每個模型+每個類別的分類指標 | Per-class 分類分析 |
| `results/perclass_metrics_detailed.json` | 完整的 JSON 格式數據 | 程式化分析 |

## 🎯 建議的工作流程

### 選項 A：先使用整體指標

如果您現在就需要結果：

1. **使用現有的整體指標**
   ```python
   import pandas as pd
   df = pd.read_csv('results/comprehensive_metrics.csv')
   print(df)
   ```

2. **在論文中使用**
   ```latex
   \input{results/comprehensive_table.tex}
   ```

3. **稍後補充 per-class 數據**
   - 在有時間時運行提取
   - 添加 per-class 分析作為補充

### 選項 B：先提取 Per-Class（推薦）

如果您需要完整的 per-class 分析：

1. **運行提取**
   ```powershell
   .\run_perclass_extraction.ps1
   ```

2. **等待 30-60 分鐘**
   - 可以去做其他工作
   - 腳本會自動處理所有模型

3. **使用完整數據**
   ```python
   # 整體指標
   overall = pd.read_csv('results/comprehensive_metrics.csv')
   
   # Per-class 檢測
   det_perclass = pd.read_csv('results/perclass_detection_metrics.csv')
   
   # Per-class 分類
   cls_perclass = pd.read_csv('results/perclass_classification_metrics.csv')
   ```

## 📖 相關文檔

### 主要文檔
- **`PERCLASS_METRICS_GUIDE.md`** ⭐ - Per-class 指標完整指南
- **`COMPARISON_RESULTS_SUMMARY.md`** - 整體結果分析
- **`AGGREGATE_RESULTS_README.md`** - 聚合工具使用說明

### 腳本文件
- `aggregate_with_perclass.py` - 整體指標聚合（已運行）
- `extract_perclass_metrics.py` - Per-class 提取（需要運行）
- `check_available_perclass_data.py` - 數據可用性檢查
- `run_perclass_extraction.ps1` - Per-class 提取運行腳本

## 💡 快速示例

### 查看整體檢測指標

```python
import pandas as pd

df = pd.read_csv('results/comprehensive_metrics.csv')

# 最佳檢測模型（按 mAP@0.5 排序）
print(df.sort_values('mAP_0.5', ascending=False)[['model_type', 'architecture', 'precision', 'recall', 'mAP_0.5']])
```

### 查看整體分類指標

```python
import pandas as pd

df = pd.read_csv('results/comprehensive_metrics.csv')

# 最佳分類模型（按準確率排序）
print(df.sort_values('cls_accuracy', ascending=False)[['model_type', 'architecture', 'cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1_score']])
```

### 查看 Per-Class 檢測（提取後）

```python
import pandas as pd

df = pd.read_csv('results/perclass_detection_metrics.csv')

# A4C 類別的檢測性能
print(df[df['Class'] == 'A4C'].sort_values('mAP@0.5', ascending=False))
```

### 查看 Per-Class 分類（提取後）

```python
import pandas as pd

df = pd.read_csv('results/perclass_classification_metrics.csv')

# PSAX 類別的分類性能
print(df[df['Class'] == 'PSAX'].sort_values('Accuracy', ascending=False))
```

## ❓ 常見問題

### Q: 我現在能用整體指標嗎？
**A**: 可以！`results/comprehensive_metrics.csv` 和 `comprehensive_table.tex` 已經可以使用了。

### Q: Per-class 數據一定要提取嗎？
**A**: 取決於您的需求。如果論文需要每個類別的詳細分析（如 A4C vs PSAX 的性能對比），就需要提取。

### Q: 提取過程可以暫停嗎？
**A**: 可以中斷（Ctrl+C），但已提取的數據會丟失。建議一次性完成。

### Q: 可以只提取部分模型嗎？
**A**: 可以修改 `extract_perclass_metrics.py` 中的 `YOLOV5C_MODELS` 列表。

### Q: 提取失敗了怎麼辦？
**A**: 檢查：
1. 模型權重是否存在（`yolov5c/thesis results/*/weights/best.pt`）
2. 數據集路徑是否正確（`regurgitationV1/data.yaml` 等）
3. GPU 記憶體是否足夠

## 🚀 立即開始

### 現在就想看結果？
```powershell
# 查看整體指標
python -c "import pandas as pd; print(pd.read_csv('results/comprehensive_metrics.csv'))"

# 查看 LaTeX 表格
cat results\comprehensive_table.tex
```

### 需要 Per-Class 數據？
```powershell
# 運行提取（需要 30-60 分鐘）
.\run_perclass_extraction.ps1
```

### 先了解一下？
```powershell
# 閱讀完整指南
cat PERCLASS_METRICS_GUIDE.md

# 檢查可用數據
python check_available_perclass_data.py
```

---

## 📝 總結

✅ **整體指標**：已完成，可以立即使用  
⚠️ **Per-Class 指標**：需要運行 `extract_perclass_metrics.py`（30-60 分鐘）

選擇最適合您需求的方案，開始分析吧！


