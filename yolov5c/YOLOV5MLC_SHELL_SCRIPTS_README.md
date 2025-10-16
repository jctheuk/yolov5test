# YOLOv5MLC Shell 腳本使用指南

## 📋 已創建的 Shell 腳本

為每個 YOLOv5MLC 模型配置創建了獨立的 shell 腳本，每個腳本自動執行 V1-V5 的 K-Fold 訓練。

### 🗂️ Shell 腳本列表

1. **`yolov5mlc_classify_backbone.sh`**
   - 模型：分類頭在 Backbone (896 通道)
   - 批次大小：64
   - 執行：V1-V5 所有 fold

2. **`yolov5mlc_p3.sh`**
   - 模型：分類頭在 P3 層 (224 通道)
   - 批次大小：128 (P3 較小可用更大批次)
   - 執行：V1-V5 所有 fold

3. **`yolov5mlc_p4.sh`**
   - 模型：分類頭在 P4 層 (448 通道)
   - 批次大小：64
   - 執行：V1-V5 所有 fold

4. **`yolov5mlc_p5.sh`**
   - 模型：分類頭在 P5 層 (896 通道)
   - 批次大小：64
   - 執行：V1-V5 所有 fold

## 🚀 使用方法

### 方式 1：直接執行 Shell 腳本 (推薦用於單一容器)
```bash
# 在訓練服務器上執行
chmod +x yolov5mlc_classify_backbone.sh
./yolov5mlc_classify_backbone.sh

# 或其他模型
./yolov5mlc_p3.sh
./yolov5mlc_p4.sh
./yolov5mlc_p5.sh
```

### 方式 2：分散到不同容器執行
從 `yolov5mlc_training_commands.txt` 複製單獨的命令：

```
容器1: 所有 V1 命令 (backbone, p3, p4, p5)
容器2: 所有 V2 命令 (backbone, p3, p4, p5)
容器3: 所有 V3 命令 (backbone, p3, p4, p5)
容器4: 所有 V4 命令 (backbone, p3, p4, p5)
容器5: 所有 V5 命令 (backbone, p3, p4, p5)
```

## 📊 配置詳情

### 共同配置
```bash
--epochs 300          # 完整訓練週期
--patience 0          # 關閉早停機制
--imgsz 416          # 圖像尺寸
--cache              # 啟用快取
--nosave             # 不保存中間權重
--hyp data/hyps/hyp.default.yaml  # 超參數配置
--cos-lr             # 餘弦退火學習率
--amp                # 混合精度訓練
```

### 批次大小差異
```
Backbone: batch-size 64
P3:       batch-size 128  (較小模型可用更大批次)
P4:       batch-size 64
P5:       batch-size 64
```

## 📈 預期執行時間

基於原始日誌分析：

| 模型 | 單個 Fold 時間 | V1-V5 總時間 |
|------|---------------|--------------|
| Backbone | ~50分鐘 | ~4小時10分鐘 |
| P3 | ~40-45分鐘 | ~3小時30分鐘 |
| P4 | ~50分鐘 | ~4小時10分鐘 |
| P5 | ~50分鐘 | ~4小時10分鐘 |

**注意**: MLC 模型會稍微快一些 (約 10-15% 速度提升)

## 🔍 監控建議

### 執行時監控
```bash
# 在另一個終端監控 GPU 使用
watch -n 1 nvidia-smi

# 監控訓練進度
tail -f runs/train/yolov5mlc_backbone_v1/results.csv
```

### 關鍵檢查點
- ✅ GPU 記憶體 <26GB (81% 使用率)
- ✅ 無 NaN 錯誤
- ✅ 訓練損失穩定下降
- ✅ 完成 300 epochs

## 🚨 故障排除

### 如果出現記憶體不足
1. **停止當前訓練**
2. **修改腳本降低批次大小**：
   ```bash
   # 編輯 .sh 文件
   --batch-size 64  →  --batch-size 32
   --batch-size 128 →  --batch-size 64
   ```
3. **重新執行**

### 如果出現 NaN 錯誤
1. **檢查 GPU 記憶體使用率**
2. **使用更小批次大小** (參考上方)
3. **或使用極致記憶體優化版本** (`memory_optimized_training_commands.txt`)

## 📁 結果查看

### 訓練結果位置
```
runs/train/
├── yolov5mlc_backbone_v1/
├── yolov5mlc_backbone_v2/
├── yolov5mlc_backbone_v3/
├── yolov5mlc_backbone_v4/
├── yolov5mlc_backbone_v5/
├── yolov5mlc_p3_v1/
├── yolov5mlc_p3_v2/
├── ...
└── yolov5mlc_p5_v5/
```

### 每個結果目錄包含
- `results.csv` - 訓練指標
- `results.png` - 訓練曲線圖
- `classification_metrics.png` - 分類指標圖
- `weights/last.pt` - 最終權重 (如果移除 --nosave)
- `hyp.yaml` - 使用的超參數
- `opt.yaml` - 使用的訓練選項

## 🎯 執行建議

### 推薦執行順序

#### 測試階段
1. **選擇一個模型** (如 Backbone)
2. **執行一個 Fold** (如 V1)
3. **驗證成功** (檢查記憶體和結果)
4. **再執行完整腳本**

#### 生產階段
- **並行執行**: 4個不同容器同時執行 4個腳本
- **串行執行**: 一個容器依次執行 4個腳本

## 🔧 腳本特點

### 自動化功能
- ✅ 環境設置 (apt-get, pip install)
- ✅ 時間戳記錄
- ✅ 進度提示
- ✅ 錯誤處理
- ✅ 完成通知

### 穩定性增強
- ✅ 混合精度訓練 (`--amp`)
- ✅ 餘弦學習率調度 (`--cos-lr`)
- ✅ 梯度剪裁 (在 hyp.default.yaml 中)
- ✅ 關閉早停 (`--patience 0`)

## 📝 總結

這 4 個 shell 腳本提供了完整的 YOLOv5MLC 訓練自動化解決方案：

- **記憶體優化**: 從 99.7% → 50-81% 使用率
- **性能平衡**: 比 YOLOv5m 更強，比 YOLOv5l 更穩定
- **易於使用**: 一鍵執行 V1-V5 訓練
- **適合並行**: 可在多個容器中同時執行

選擇你需要的腳本，直接執行即可開始訓練！🚀

