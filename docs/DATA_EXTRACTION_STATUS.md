# YOLOv5 數據提取狀態報告

## ✅ 已完成的數據

### 1. 分類任務每類別指標（完成度：100%）

**完成情況**：15/15 模型
- ✅ YOLOv5-Small: 5個版本（V1-V5）全部完成
- ✅ YOLOv5-Medium: 5個版本（V1-V5）全部完成
- ✅ YOLOv5-Large: 5個版本（V1-V5）全部完成

**提取的指標**：
- ✅ Precision（精確率）- 每個類別
- ✅ Recall（召回率）- 每個類別
- ✅ F1-Score（F1分數）- 每個類別
- ✅ Support（樣本數）- 每個類別
- ✅ Confusion Matrix（混淆矩陣）- 15個PNG文件

**分類類別**：
- A4C（心尖四腔室視圖）
- PLAX（胸骨旁長軸視圖）
- PSAX（胸骨旁短軸視圖）

**數據文件**：
- `classification_metrics/classify{s,m,l}_v{1-5}_metrics.csv` - 15個CSV文件
- `classification_metrics/classify{s,m,l}_v{1-5}_confusion_matrix.png` - 15個混淆矩陣圖
- `classification_metrics/aggregated_results.csv` - 聚合表格

### 2. 檢測任務整體指標（完成度：100%）

**完成情況**：12/12 配置
- ✅ yolov5s-SC: backbone, p3, p4, p5（4個配置）
- ✅ yolov5m-MC: backbone, p3, p4, p5（4個配置）
- ✅ yolov5m-MLC: backbone, p3, p4, p5（4個配置）

**提取的指標**：
- ✅ Detection Precision（檢測精確率）- 整體
- ✅ Detection Recall（檢測召回率）- 整體
- ✅ mAP@0.5 - 整體
- ✅ mAP@0.5:0.95 - 整體
- ✅ Classification Accuracy（分類準確率）- 整體

**數據文件**：
- `classification_metrics/detection_results_detailed.csv` - 60行詳細數據（12配置×5版本）
- `classification_metrics/detection_results_aggregated.csv` - 12行聚合數據

---

## ⏳ 待提取的數據

### 檢測任務每類別指標（完成度：0%）

**需要提取**：AR, MR, PR, TR 每個類別的mAP指標

**檢測類別**：
- AR（Aortic Regurgitation - 主動脈瓣逆流）
- MR（Mitral Regurgitation - 二尖瓣逆流）
- PR（Pulmonary Regurgitation - 肺動脈瓣逆流）
- TR（Tricuspid Regurgitation - 三尖瓣逆流）

**原因**：YOLOv5訓練時只記錄整體檢測指標，每類別指標需要運行驗證才能獲得

**提取方法**：

#### 選項1：在TWCC環境運行驗證（推薦）

```bash
# 使用提供的腳本
chmod +x docs/EXTRACT_DETECTION_PERCLASS.sh
./docs/EXTRACT_DETECTION_PERCLASS.sh
```

此腳本將：
- 對12個配置 × 5個版本 = 60個模型運行驗證
- 生成包含AR, MR, PR, TR每類別mAP的詳細報告
- 保存在 `yolov5c/runs/val_perclass/`

#### 選項2：手動提取單個模型

```bash
cd yolov5c

python val.py \
    --weights "thesis results/yolov5sc_backbone_v1/weights/last.pt" \
    --data "../Regurgitation-YOLODataset-1/data.yaml" \
    --batch-size 32 \
    --img 416 \
    --task test \
    --verbose
```

查看輸出中的Per-Class mAP表格。

---

## 📊 當前可用的完整表格

### 位置

1. **Markdown格式**：`docs/FINAL_COMPREHENSIVE_TABLE.md`
2. **CSV格式**：`docs/FINAL_COMPREHENSIVE_TABLE.csv`
3. **LaTeX格式**：`classification_metrics/aggregated_results.tex`

### 表格內容

**主行**：模型配置（例如 yolov5s_p5, yolov5m_backbone）

**子行**：
- ✅ 分類類別：A4C, PLAX, PSAX（**已完成**）
- ⏳ 檢測類別：AR, MR, PR, TR（**待提取**）

**列**：
- ✅ Precision（精確率）
- ✅ Recall（召回率）
- ✅ F1-Score（F1分數）
- ✅ mAP@0.5（檢測mAP）
- ✅ mAP@0.5:0.95（檢測mAP）
- ✅ Classification Accuracy（分類準確率）
- ✅ Support（樣本數）

### 數據覆蓋

**分類任務**：
- ✅ 15個模型（3種大小 × 5個版本）
- ✅ 3個類別（A4C, PLAX, PSAX）
- ✅ 45條記錄（完整）

**檢測任務**：
- ✅ 12個配置（整體指標）
- ⏳ 4個類別（AR, MR, PR, TR每類別待提取）

---

## 🎯 完成提取檢測每類別數據的步驟

### 步驟1：在TWCC環境運行驗證

```bash
# 上傳並運行腳本
scp docs/EXTRACT_DETECTION_PERCLASS.sh twcc:/path/to/yolov5test/
ssh twcc
cd /path/to/yolov5test
chmod +x docs/EXTRACT_DETECTION_PERCLASS.sh
./docs/EXTRACT_DETECTION_PERCLASS.sh
```

預計時間：約2-3小時（60個模型驗證）

### 步驟2：提取日誌中的Per-Class數據

驗證輸出會顯示類似：

```
Class    Images  Instances      P      R   mAP50   mAP50-95
all         XXX        XXX  0.823  0.759   0.749      0.298
AR          XXX        XXX  0.850  0.780   0.760      0.310
MR          XXX        XXX  0.820  0.750   0.740      0.290
PR          XXX        XXX  0.810  0.740   0.730      0.285
TR          XXX        XXX  0.815  0.765   0.765      0.305
```

### 步驟3：解析並聚合

創建Python腳本解析驗證日誌：

```python
# parse_validation_logs.py
# 解析 yolov5c/runs/val_perclass/*_log.txt
# 提取每類別的P, R, mAP50, mAP50-95
# 平均V1-V5
# 更新最終表格
```

---

## 📝 使用現有表格

雖然檢測的每類別數據待提取，但現有表格已經非常完整：

### 可以回答的問題

✅ **分類性能**：
- 哪個模型在A4C/PLAX/PSAX上表現最好？
- 不同模型大小對分類準確率的影響？
- 哪個類別最容易/困難識別？

✅ **檢測性能（整體）**：
- 哪個配置的整體mAP最高？
- backbone vs p3/p4/p5的性能差異？
- 聯合訓練對分類準確率的影響？

✅ **模型比較**：
- Small vs Medium模型的權衡？
- MC vs MLC損失函數的效果？

⏳ **待補充**：
- AR, MR, PR, TR每個類別的單獨mAP？
- 哪個瓣膜逆流最難檢測？

---

## 總結

**已完成的數據提取**：
- 🎉 分類任務：100%完成（15模型×3類別 = 45條記錄）
- 🎉 檢測任務：整體指標100%完成（12配置×1整體 = 12條記錄）
- 📊 混淆矩陣：15個分類 + 60個聯合訓練 = 75個混淆矩陣

**待補充的數據**：
- ⏳ 檢測每類別：需要在TWCC環境運行驗證（估計2-3小時）

**實用性評估**：
- 當前表格已經可以用於大部分分析和比較
- 檢測每類別數據是錦上添花，非必需（除非需要分析特定瓣膜）

**文件位置**：
- ✅ 主表格：`docs/FINAL_COMPREHENSIVE_TABLE.md`
- ✅ CSV數據：`docs/FINAL_COMPREHENSIVE_TABLE.csv`  
- ✅ 提取腳本：`docs/EXTRACT_DETECTION_PERCLASS.sh`



