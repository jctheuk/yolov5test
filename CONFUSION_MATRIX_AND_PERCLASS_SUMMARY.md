# 混淆矩陣和 Per-Class 指標總結

## ✅ 已完成並可用

### 1. 分類混淆矩陣（Classification Confusion Matrices）

**總數**：60 個混淆矩陣
- 12 個模型 × 5 個版本 (v1-v5)
- 每個矩陣顯示 A4C 和 PSAX 類別的分類性能

**位置**：
- **原始位置**：`yolov5c/thesis results/{model_name}_{version}/classification_metrics_combined.png`
- **整理後位置**：`results/perclass_analysis/confusion_matrices/{model_name}/classification_{version}.png`

**摘要文檔**：
- `results/perclass_analysis/confusion_matrix_summary.md` - 所有混淆矩陣的索引

#### 可用的模型混淆矩陣

| 模型 | 架構 | 混淆矩陣數量 |
|------|------|------------|
| YOLOv5-SC | backbone, p3, p4, p5 | 5 × 4 = 20 |
| YOLOv5-MC | backbone, p3, p4, p5 | 5 × 4 = 20 |
| YOLOv5-MLC | backbone, p3, p4, p5 | 5 × 4 = 20 |
| **總計** | | **60** |

### 2. 整體指標（Overall Metrics）

**檢測指標**：
- ✅ Precision (P) - 整體檢測精確度
- ✅ Recall (R) - 整體檢測召回率
- ✅ mAP@0.5 - 整體平均精度
- ✅ mAP@0.5:0.95 - 整體平均精度

**分類指標**：
- ✅ Accuracy - 整體分類準確率
- ✅ Precision - 整體分類精確度
- ✅ Recall - 整體分類召回率
- ✅ F1-Score - 整體 F1 分數

**文件**：
- `results/comprehensive_metrics.csv` - CSV 格式
- `results/comprehensive_table.tex` - LaTeX 表格
- `results/perclass_analysis/perclass_summary.csv` - Per-class 摘要

## 📊 如何使用混淆矩陣

### 查看特定模型的混淆矩陣

#### 方法 1：查看原始位置
```powershell
# 查看 YOLOv5-SC backbone v1 的分類混淆矩陣
start "yolov5c\thesis results\yolov5sc_backbone_v1\classification_metrics_combined.png"
```

#### 方法 2：查看整理後的位置
```powershell
# 查看所有 YOLOv5-SC backbone 的混淆矩陣
start "results\perclass_analysis\confusion_matrices\yolov5sc_backbone"
```

### 查看摘要索引
```powershell
# 查看所有混淆矩陣的列表
cat results\perclass_analysis\confusion_matrix_summary.md
```

### 比較不同模型

```powershell
# 比較 backbone 架構的三個損失類型
start results\perclass_analysis\confusion_matrices\yolov5sc_backbone\classification_v1.png
start results\perclass_analysis\confusion_matrices\yolov5mc_backbone\classification_v1.png
start results\perclass_analysis\confusion_matrices\yolov5mlc_backbone\classification_v1.png
```

## 📈 從混淆矩陣中可以得到的 Per-Class 信息

混淆矩陣顯示了**每個類別**的性能：

### 對於 A4C 類別
- **True Positives (TP)**：正確識別為 A4C 的數量
- **False Positives (FP)**：錯誤識別為 A4C 的數量（實際是 PSAX）
- **False Negatives (FN)**：錯誤識別為 PSAX 的數量（實際是 A4C）
- **True Negatives (TN)**：正確識別為 PSAX 的數量

### 對於 PSAX 類別
- 同理可得

### 計算 Per-Class 指標

從混淆矩陣可以計算：

```
對於類別 i：
Precision_i = TP_i / (TP_i + FP_i)
Recall_i = TP_i / (TP_i + FN_i)
F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
Accuracy_i = (TP_i + TN_i) / Total
```

## 🎯 完整的 Per-Class 數據匯總

### 已有數據

#### 1. 分類 Per-Class（來自混淆矩陣）
- ✅ **A4C 分類性能** - 從 60 個混淆矩陣可視化
- ✅ **PSAX 分類性能** - 從 60 個混淆矩陣可視化

#### 2. 整體檢測和分類指標
- ✅ **整體檢測**：P, R, mAP@0.5, mAP@0.5:0.95
- ✅ **整體分類**：Accuracy, Precision, Recall, F1

### 還需要的數據（如果需要）

#### 檢測 Per-Class（需要運行驗證）
- ⚠️ **A4C 檢測**：Precision, Recall, AP@0.5, AP@0.5:0.95
- ⚠️ **PSAX 檢測**：Precision, Recall, AP@0.5, AP@0.5:0.95

如果需要檢測的 per-class 指標，需要運行：
```powershell
python yolov5c/val.py --weights <model> --data <data.yaml> --verbose
```

## 📁 文件結構

```
results/
├── comprehensive_metrics.csv          # 整體指標（所有模型）
├── comprehensive_table.tex            # LaTeX 表格
├── combined_metrics.csv               # 簡化版本
├── combined_table.tex                 # 簡化版本 LaTeX
└── perclass_analysis/
    ├── confusion_matrix_summary.md    # 混淆矩陣索引
    ├── perclass_summary.csv           # Per-class 摘要
    └── confusion_matrices/            # 所有混淆矩陣
        ├── yolov5sc_backbone/
        │   ├── classification_v1.png
        │   ├── classification_v2.png
        │   ├── classification_v3.png
        │   ├── classification_v4.png
        │   └── classification_v5.png
        ├── yolov5sc_p3/
        ├── yolov5sc_p4/
        ├── yolov5sc_p5/
        ├── yolov5mc_backbone/
        ├── yolov5mc_p3/
        ├── yolov5mc_p4/
        ├── yolov5mc_p5/
        ├── yolov5mlc_backbone/
        ├── yolov5mlc_p3/
        ├── yolov5mlc_p4/
        └── yolov5mlc_p5/
```

## 💡 使用範例

### 在論文中使用

#### 方法 1：展示單一模型的混淆矩陣
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{results/perclass_analysis/confusion_matrices/yolov5sc_backbone/classification_v1.png}
  \caption{YOLOv5-SC Backbone V1 Classification Confusion Matrix}
  \label{fig:cm_sc_backbone_v1}
\end{figure}
```

#### 方法 2：展示多個模型的比較
```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}{0.3\textwidth}
    \includegraphics[width=\textwidth]{results/perclass_analysis/confusion_matrices/yolov5sc_backbone/classification_v1.png}
    \caption{SC}
  \end{subfigure}
  \begin{subfigure}{0.3\textwidth}
    \includegraphics[width=\textwidth]{results/perclass_analysis/confusion_matrices/yolov5mc_backbone/classification_v1.png}
    \caption{MC}
  \end{subfigure}
  \begin{subfigure}{0.3\textwidth}
    \includegraphics[width=\textwidth]{results/perclass_analysis/confusion_matrices/yolov5mlc_backbone/classification_v1.png}
    \caption{MLC}
  \end{subfigure}
  \caption{Comparison of Classification Confusion Matrices}
\end{figure}
```

### 在 Python 中分析

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 讀取整體指標
df = pd.read_csv('results/comprehensive_metrics.csv')

# 查看最佳模型
best_cls = df.sort_values('cls_accuracy', ascending=False).iloc[0]
print(f"Best classification model: {best_cls['model_type']} {best_cls['architecture']}")
print(f"Accuracy: {best_cls['cls_accuracy']:.2f}%")

# 顯示對應的混淆矩陣
model_name = f"yolov5{best_cls['model_type'].split('-')[-1].lower()}_{best_cls['architecture']}"
cm_path = f"results/perclass_analysis/confusion_matrices/{model_name}/classification_v1.png"

img = Image.open(cm_path)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"{best_cls['model_type']} {best_cls['architecture']} - Best Model")
plt.show()
```

## 📊 統計摘要

### 分類混淆矩陣統計

| 類型 | 數量 | 狀態 |
|------|------|------|
| YOLOv5c 分類混淆矩陣 | 60 | ✅ 可用 |
| YOLOv5c 檢測混淆矩陣 | 0 | ⚠️ 未生成 |
| YOLOv5 Original 混淆矩陣 | 有（在 val-cls 中） | ✅ 可用 |

### 模型覆蓋率

- ✅ **100% 模型**有分類混淆矩陣（12 個模型 × 5 個版本）
- ✅ **所有版本**都有完整記錄（v1, v2, v3, v4, v5）
- ✅ **所有架構**都有混淆矩陣（backbone, p3, p4, p5）

## 🎓 論文撰寫建議

### 1. 整體性能比較
使用：`results/comprehensive_table.tex`

### 2. 最佳模型展示
使用：最佳模型的混淆矩陣
- YOLOv5-M (Original): 98.25% 準確率
- YOLOv5-SC backbone: 97.45% 準確率 + 最佳檢測性能

### 3. 架構比較
展示同一損失類型下不同架構的混淆矩陣：
- backbone vs p3 vs p4 vs p5

### 4. 損失策略比較
展示同一架構下不同損失的混淆矩陣：
- SC vs MC vs MLC

### 5. Per-Class 分析
從混淆矩陣中提取 A4C 和 PSAX 的個別性能

## ❓ 常見問題

### Q: 混淆矩陣顯示什麼？
A: 顯示 A4C 和 PSAX 兩個類別的分類結果，包括正確和錯誤分類的數量。

### Q: 為什麼沒有檢測的混淆矩陣？
A: 檢測任務通常不使用傳統的混淆矩陣，而是使用 mAP、AP 等指標。如果需要 per-class 檢測指標，需要運行詳細驗證。

### Q: 如何比較不同版本的性能？
A: 查看同一模型下的 v1-v5 混淆矩陣，觀察分類錯誤的模式變化。

### Q: 可以自動計算 per-class 指標嗎？
A: 可以從混淆矩陣圖像中提取數據，或直接從訓練日誌中解析。目前提供的是視覺化混淆矩陣。

## 🚀 快速開始

### 查看所有混淆矩陣
```powershell
# 打開混淆矩陣目錄
start results\perclass_analysis\confusion_matrices
```

### 查看最佳模型
```powershell
# YOLOv5-SC backbone (最佳檢測和分類平衡)
start results\perclass_analysis\confusion_matrices\yolov5sc_backbone

# YOLOv5-MC p3 (最佳分類準確率)
start results\perclass_analysis\confusion_matrices\yolov5mc_p3
```

### 查看摘要
```powershell
# 查看混淆矩陣索引
cat results\perclass_analysis\confusion_matrix_summary.md

# 查看 per-class 摘要
python -c "import pandas as pd; print(pd.read_csv('results/perclass_analysis/perclass_summary.csv'))"
```

## 📚 相關文檔

- `COMPARISON_RESULTS_SUMMARY.md` - 整體結果詳細分析
- `CURRENT_STATUS_AND_NEXT_STEPS.md` - 當前狀態和後續步驟
- `PERCLASS_METRICS_GUIDE.md` - Per-class 指標提取指南
- `results/perclass_analysis/confusion_matrix_summary.md` - 混淆矩陣完整索引

---

## 總結

✅ **已完成**：
- 60 個分類混淆矩陣（每個類別：A4C, PSAX）
- 整體檢測和分類指標
- 完整的混淆矩陣索引和組織

✅ **可立即使用**：
- 所有混淆矩陣圖像
- CSV 和 LaTeX 格式的指標表格
- 完整的文檔和使用指南

🎯 **您現在擁有完整的 per-class 視覺化數據，可以直接用於論文和分析！**


